#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x00000000000002e8,0x0000004001010002,0x00000000000002a8\n"
".quad 0x0000000000000000,0x0000004600010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007900be0002,0x0000000000000000,0x0000000000000238,0x00000000000000f8\n"
".quad 0x0038004000460546,0x0001000500400002,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x7466752e766e2e00\n"
".quad 0x2e007972746e652e,0x006f666e692e766e,0x665f67756265642e,0x732e0000656d6172\n"
".quad 0x0062617472747368,0x006261747274732e,0x006261746d79732e,0x5f6261746d79732e\n"
".quad 0x6e2e0078646e6873,0x6e652e7466752e76,0x2e766e2e00797274,0x65642e006f666e69\n"
".quad 0x6d6172665f677562,0x0000000000000065,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000300000001,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000040,0x000000000000004d,0x0000000000000000,0x0000000000000001\n"
".quad 0x0000000000000000,0x000000030000000b,0x0000000000000000,0x0000000000000000\n"
".quad 0x000000000000008d,0x000000000000004d,0x0000000000000000,0x0000000000000001\n"
".quad 0x0000000000000000,0x0000000200000013,0x0000000000000000,0x0000000000000000\n"
".quad 0x00000000000000e0,0x0000000000000018,0x0000000100000002,0x0000000000000008\n"
".quad 0x0000000000000018,0x0000000100000040,0x0000000000000000,0x0000000000000000\n"
".quad 0x00000000000000f8,0x0000000000000000,0x0000000000000000,0x0000000000000001\n"
".quad 0x0000000000000000,0x0000000500000006,0x0000000000000238,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000070,0x0000000000000070,0x0000000000000008\n"
".quad 0x0000000500000001,0x0000000000000238,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000070, 0x0000000000000070, 0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[95];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif