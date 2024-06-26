#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x0000000000000338,0x0000004001010002,0x00000000000002f8\n"
".quad 0x0000000000000000,0x0000004600010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007900be0002,0x0000000000000000,0x0000000000000288,0x0000000000000148\n"
".quad 0x0038004000460546,0x0001000500400002,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x7466752e766e2e00\n"
".quad 0x2e007972746e652e,0x006f666e692e766e,0x6c6c61632e766e2e,0x6e2e006870617267\n"
".quad 0x746f746f72702e76,0x68732e0000657079,0x2e00626174727473,0x2e00626174727473\n"
".quad 0x2e006261746d7973,0x735f6261746d7973,0x766e2e0078646e68,0x746e652e7466752e\n"
".quad 0x692e766e2e007972,0x2e766e2e006f666e,0x706172676c6c6163,0x72702e766e2e0068\n"
".quad 0x00657079746f746f,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0004000300000040,0x0000000000000000,0x0000000000000000,0xffffffff00000000\n"
".quad 0xfffffffe00000000,0xfffffffd00000000,0xfffffffc00000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000300000001\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000040,0x000000000000005c\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x000000030000000b\n"
".quad 0x0000000000000000,0x0000000000000000,0x000000000000009c,0x000000000000005c\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x0000000200000013\n"
".quad 0x0000000000000000,0x0000000000000000,0x00000000000000f8,0x0000000000000030\n"
".quad 0x0000000200000002,0x0000000000000008,0x0000000000000018,0x7000000100000040\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000128,0x0000000000000020\n"
".quad 0x0000000000000005,0x0000000000000004,0x0000000000000008,0x0000000500000006\n"
".quad 0x0000000000000288,0x0000000000000000,0x0000000000000000,0x0000000000000070\n"
".quad 0x0000000000000070,0x0000000000000008,0x0000000500000001,0x0000000000000288\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000070,0x0000000000000070\n"
".quad 0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[105];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 2, fatbinData, (void**)__cudaPrelinkedFatbins };
#ifdef __cplusplus
}
#endif
