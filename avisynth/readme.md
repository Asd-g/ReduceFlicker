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
	Clip must be in Y/YUV 8..32-bit format.

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
	Default value is false.

#### opt:
	Controls which cpu optimizations are used.
	Currently, this filter has four routines.

	0 - Use C++ routine.
	1 - Use SSE2/SSE routine. If cpu does not have SSE2, fallback to 0.
	2 - Use SSE4.1/SSE2/SSE routine. If cpu does not have SSE4.1, fallback to 1.
    3 (default) - Use AVX2/AVX routine. If cpu does not have AVX2, fallback to 2.
                      
#### raccess:
    When the previous and next frames are accessed.
    
    True (default) - the next frames are accessed fisrt.
    False - the previous frame are accessed first.
    
#### luma:
    Whether luma plane will be processed or not. If set this to false, luma plane will be garbage.
    Default value is true.

### Lisence:
	GPLv2 or later.

### Source code:
	https://github.com/chikuzen/ReduceFlicker/
