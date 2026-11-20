# Run in a Vivado Tcl console with a project open.
set dino_root [file dirname [file normalize [info script]]]
set dino_rtl [file join $dino_root rtl]
set dino_fileset [current_fileset -srcset]
set dino_header [file join $dino_rtl addr_map.vh]
set dino_roms [lsort [glob -directory [file join $dino_root mem] *.mem]]

add_files -norecurse -fileset $dino_fileset [concat \
    [lsort [glob -directory $dino_rtl *.v]] [list $dino_header] $dino_roms]
set_property FILE_TYPE {Verilog Header} [get_files $dino_header]
set_property FILE_TYPE {Memory Initialization Files} [get_files $dino_roms]

set dino_includes [get_property INCLUDE_DIRS $dino_fileset]
if {[lsearch -exact $dino_includes $dino_rtl] < 0} {
    lappend dino_includes $dino_rtl
    set_property INCLUDE_DIRS $dino_includes $dino_fileset
}
