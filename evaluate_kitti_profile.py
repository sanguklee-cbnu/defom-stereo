from __future__ import annotations

import argparse
import inspect
import logging
import math
from typing import Any


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, learnable


def custom_get_danv2_io_size(h, w, nds, factor=3.5, max_i_size=2688, multiple_of=14):
    oh, ow = h // (2 ** nds), w // (2 ** nds)

    ih = factor * oh
    iw = factor * ow
    ih = math.ceil(ih / multiple_of) * multiple_of
    iw = math.ceil(iw / multiple_of) * multiple_of

    max_i_size = math.floor(max_i_size / multiple_of) * multiple_of
    if ih <= max_i_size and iw <= max_i_size:
        return ih, iw, oh, ow

    factor_h = max_i_size / ih
    factor_w = max_i_size / iw
    if factor_w > factor_h:
        new_ih = max_i_size
        new_iw = math.ceil(factor_h * iw / multiple_of) * multiple_of
    else:
        new_iw = max_i_size
        new_ih = math.ceil(factor_w * ih / multiple_of) * multiple_of
    return new_ih, new_iw, oh, ow


def patch_danv2_io_size(da_factor):
    def patched(h, w, nds, max_i_size=2688, multiple_of=14):
        return custom_get_danv2_io_size(
            h,
            w,
            nds,
            factor=da_factor,
            max_i_size=max_i_size,
            multiple_of=multiple_of,
        )

    return patched


def timed_call(fn, *args, use_amp=False, amp_dtype=None, **kwargs):
    import torch

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if use_amp:
        with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype or torch.float16):
            out = fn(*args, **kwargs)
    else:
        out = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


def benchmark(fn, args=(), kwargs=None, warmup=5, iters=20, use_amp=False, amp_dtype=None):
    import torch

    kwargs = kwargs or {}
    with torch.inference_mode():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype or torch.float16):
                    _ = fn(*args, **kwargs)
            else:
                _ = fn(*args, **kwargs)
        torch.cuda.synchronize()

        last_out = None
        times = []
        for _ in range(iters):
            last_out, ms = timed_call(fn, *args, use_amp=use_amp, amp_dtype=amp_dtype, **kwargs)
            times.append(ms)
    return last_out, times


def build_da_input(model, image1, image2, n_downsample, da_factor):
    import torch
    import torch.nn.functional as F

    image1_n = ((image1 - model.mean) / model.std).contiguous()
    image2_n = ((image2 - model.mean) / model.std).contiguous()
    bs, _, h, w = image1_n.shape
    ih, iw, oh, ow = custom_get_danv2_io_size(h, w, n_downsample, factor=da_factor)
    x = torch.cat([image1_n, image2_n], dim=0)
    x = F.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=True)
    return x, bs, ih, iw, oh, ow


def profile_depth_anything(model, image1, image2, args):
    da = model.defomencoder.depth_anything
    x_da, _, ih, iw, oh, ow = build_da_input(model, image1, image2, args.n_downsample, args.da_factor)
    patch_h, patch_w = ih // 14, iw // 14

    _, da_total_times = benchmark(
        da,
        args=(x_da, oh, ow),
        warmup=args.profile_warmup,
        iters=args.profile_iters,
        use_amp=args.mixed_precision,
        amp_dtype=args.amp_dtype,
    )

    vit_fn = lambda x: da.pretrained.get_intermediate_layers(
        x, da.intermediate_layer_idx[da.encoder], return_class_token=True
    )
    features, vit_times = benchmark(
        vit_fn,
        args=(x_da,),
        warmup=args.profile_warmup,
        iters=args.profile_iters,
        use_amp=args.mixed_precision,
        amp_dtype=args.amp_dtype,
    )

    feat_fn = lambda feats: da.depth_feat(feats, patch_h, patch_w, oh, ow)
    _, dptfeat_times = benchmark(
        feat_fn,
        args=(features,),
        warmup=args.profile_warmup,
        iters=args.profile_iters,
        use_amp=args.mixed_precision,
        amp_dtype=args.amp_dtype,
    )

    head_fn = lambda feats: da.depth_head(feats, patch_h, patch_w, oh, ow)
    _, dpthead_times = benchmark(
        head_fn,
        args=(features,),
        warmup=args.profile_warmup,
        iters=args.profile_iters,
        use_amp=args.mixed_precision,
        amp_dtype=args.amp_dtype,
    )

    return {
        "x_da_shape": tuple(x_da.shape),
        "oh": oh,
        "ow": ow,
        "num_blocks": len(da.pretrained.blocks),
        "da_ms": avg(da_total_times),
        "vit_ms": avg(vit_times),
        "feat_ms": avg(dptfeat_times),
        "head_ms": avg(dpthead_times),
    }


def validate_kitti_profile(model, args):
    import numpy as np
    import torch
    import core.stereo_datasets as datasets
    from core.utils.utils import InputPadder

    split = args.kitti_split
    val_dataset = datasets.KITTI({}, split=split, image_set="training", is_eval=True)
    torch.backends.cudnn.benchmark = True

    out_list = []
    epe_list = []
    full_latency_ms = []

    da_ms_list = []
    vit_ms_list = []
    feat_ms_list = []
    head_ms_list = []

    profile_count = 0
    first_profile_meta: dict[str, Any] | None = None

    for val_id in range(len(val_dataset)):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with torch.inference_mode():
            disp_pr, latency_ms = timed_call(
                model,
                image1,
                image2,
                iters=args.valid_iters,
                scale_iters=args.scale_iters,
                test_mode=True,
                use_amp=False,
            )

        if val_id >= args.latency_start:
            full_latency_ms.append(latency_ms)

        if (
            val_id >= args.profile_start
            and profile_count < args.profile_limit
        ):
            prof = profile_depth_anything(model, image1, image2, args)
            da_ms_list.append(prof["da_ms"])
            vit_ms_list.append(prof["vit_ms"])
            feat_ms_list.append(prof["feat_ms"])
            head_ms_list.append(prof["head_ms"])
            profile_count += 1

            if first_profile_meta is None:
                first_profile_meta = {
                    "full_input": tuple(image1.shape),
                    "da_input": prof["x_da_shape"],
                    "lowres": (prof["oh"], prof["ow"]),
                    "num_blocks": prof["num_blocks"],
                }

        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)
        epe_flattened = epe.flatten()
        val = valid.flatten() >= 0.5

        if np.isnan(epe_flattened[val].mean().item()):
            logging.warning("NaN metric at sample %d, skipping metric accumulation", val_id + 1)
            continue

        out = epe_flattened > 3.0
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()

        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                "KITTI%s Iter %d out of %d. EPE %.4f Out3 %.4f. Latency: %.2f ms",
                split,
                val_id + 1,
                len(val_dataset),
                image_epe,
                image_out,
                latency_ms,
            )

        epe_list.append(image_epe)
        out_list.append(out[val].cpu().numpy())

    epe = float(np.mean(np.array(epe_list)))
    out3 = float(100.0 * np.mean(np.concatenate(out_list)))

    print()
    print(f"Validation KITTI{split}")
    print(f"  EPE              : {epe:.6f}")
    print(f"  Out3 (%)         : {out3:.6f}")
    print(f"  Full latency (ms): {avg(full_latency_ms):.2f}")

    if first_profile_meta is not None:
        da_ms = avg(da_ms_list)
        vit_ms = avg(vit_ms_list)
        feat_ms = avg(feat_ms_list)
        head_ms = avg(head_ms_list)

        print()
        print("DepthAnything Profile")
        print(f"  full input       : {first_profile_meta['full_input']}")
        print(f"  DA input         : {first_profile_meta['da_input']}")
        print(f"  lowres grid      : {first_profile_meta['lowres']}")
        print(f"  num blocks       : {first_profile_meta['num_blocks']}")
        print(f"  DepthAnything ms : {da_ms:.2f}")
        print(f"  ViT ms           : {vit_ms:.2f}")
        print(f"  DPTFeat ms       : {feat_ms:.2f}")
        print(f"  DPTHead ms       : {head_ms:.2f}")
        print(f"  Isolated sum ms  : {vit_ms + feat_ms + head_ms:.2f}")
        print(f"  ViT share        : {100.0 * vit_ms / da_ms:.1f}% of DA")
        print(f"  DPTFeat share    : {100.0 * feat_ms / da_ms:.1f}% of DA")
        print(f"  DPTHead share    : {100.0 * head_ms / da_ms:.1f}% of DA")
        print(f"  profile samples  : {profile_count}")

    return {
        f"kitti{split}-epe": epe,
        f"kitti{split}-out3": out3,
        "full-latency-ms": avg(full_latency_ms),
        "da-ms": avg(da_ms_list),
        "vit-ms": avg(vit_ms_list),
        "dptfeat-ms": avg(feat_ms_list),
        "dpthead-ms": avg(head_ms_list),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", required=True, help="restore checkpoint")
    parser.add_argument("--kitti_split", default="15", choices=["12", "15"])
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision inside DEFOM")
    parser.add_argument("--amp_dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--scale_iters", type=int, default=8)
    parser.add_argument("--latency_start", type=int, default=50, help="start averaging full latency after this sample index")
    parser.add_argument("--profile_start", type=int, default=50, help="start DA breakdown profiling at this sample index")
    parser.add_argument("--profile_limit", type=int, default=20, help="number of samples to run DA breakdown on")
    parser.add_argument("--profile_warmup", type=int, default=5)
    parser.add_argument("--profile_iters", type=int, default=20)
    parser.add_argument("--da_factor", type=float, default=3.5)

    parser.add_argument("--dinov2_encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--idepth_scale", type=float, default=0.5)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)
    parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--scale_list", type=float, nargs="+", default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--scale_corr_radius", type=int, default=2)
    parser.add_argument("--n_downsample", type=int, default=2, choices=[2, 3])
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"])
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import core.defom_stereo as defom_stereo_module
    from core.defom_stereo import DEFOMStereo

    args.amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logging.info("Loading checkpoint...")
    orig_get_danv2_io_size = defom_stereo_module.get_danv2_io_size
    if "danv2_io_factor" in inspect.signature(orig_get_danv2_io_size).parameters:
        args.danv2_io_factor = args.da_factor
    else:
        defom_stereo_module.get_danv2_io_size = patch_danv2_io_size(args.da_factor)

    try:
        model = DEFOMStereo(args)
        checkpoint = torch.load(args.restore_ckpt, map_location="cuda")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        model.cuda().eval()

        logging.info("Done loading checkpoint")
        _, learnable = count_parameters(model)
        print(f"The model has {learnable / 1e6:.2f}M learnable parameters.")
        print(f"encoder           : {args.dinov2_encoder}")
        print(f"checkpoint        : {args.restore_ckpt}")
        print(f"kitti_split       : {args.kitti_split}")
        print(f"mixed_precision   : {args.mixed_precision}")
        print(f"amp_dtype         : {'fp16' if args.amp_dtype == torch.float16 else 'bf16'}")
        print(f"da_factor         : {args.da_factor}")
        print(f"valid_iters       : {args.valid_iters}")
        print(f"scale_iters       : {args.scale_iters}")
        print(f"latency_start     : {args.latency_start}")
        print(f"profile_start     : {args.profile_start}")
        print(f"profile_limit     : {args.profile_limit}")
        print()

        validate_kitti_profile(model, args)

    finally:
        defom_stereo_module.get_danv2_io_size = orig_get_danv2_io_size


if __name__ == "__main__":
    main()
