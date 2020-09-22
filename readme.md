## ReduceFlicker for Avisynth2.6/Avisynth+
	This is a rewite of ReduceFlicker which written by Rainer Wittmann.
	This plugin has only ReduceFlicker(). ReduceFluctuation() and LockClense() are not implemented.

### Requirements:
	- avisynth2.60 / avisynth+r1576 or later.
	- WindowsVista sp2 or later.
	- Visual C++ Redistributable Packages for Visual Studio 2019.

### Syntax:
	ReduceFlicker(clip, int "strength", bool "aggressive", bool "grey", int "opt", bool "raccess", bool "luma")

#### clip:
	Clip must be in planar format.

#### strength:
	Specify the strength of ReduceFlicker. Higher values mean more aggressive operation.

	1 - makes use of 4(current + 2*previous + 1*next) frames .
	2(default) - makes use of 5(current + 2*previous + 2*next) frames.
	3 - makes use of 7(current + 3*previous + 3*next) frames.

#### aggressive:
	If set this to true, then a significantly more aggressive variant of the algorithm is selected.
	Default value is false.

#### grey:
	Whether chroma planes will be processed or not. If set this to true, chroma planes will be garbage.
	Default value is false (always false for RGB).

#### opt:
	Controls which cpu optimizations are used.
	Currently, this filter has four routines.

    -1(default) - Auto-detect.
	0 - Use C++ routine.
	1 - Use SSE2 routine.
    2 - Use AVX2 routine.
                      
#### raccess:
    When the previous and next frames are accessed.
    
    True (default) - the next frames are accessed fisrt.
    False - the previous frame are accessed first.
    
#### luma:
    Whether luma plane will be processed or not. If set this to false, luma plane will be garbage.
    Default value is true (always true for RGB).

### Lisence:
	GPLv2 or later.

### Source code:
	https://github.com/chikuzen/ReduceFlicker/
