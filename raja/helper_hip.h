/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are HIP Helper functions for initialization and error checking

#ifndef HELPER_HIP_H
#define HELPER_HIP_H

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//#include <helper_string.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Note, it is required that your SDK sample to include the proper header files, please
// refer the HIP examples for examples of the needed HIP headers, which may change depending
// on which HIP functions are used.

// HIP Runtime error messages
static const char *_hipGetErrorEnum(hipError_t error)
{
    switch (error)
    {
        case hipSuccess:
            return "hipSuccess";

        case hipErrorMissingConfiguration:
            return "hipErrorMissingConfiguration";

        case hipErrorMemoryAllocation:
            return "hipErrorMemoryAllocation";

        case hipErrorInitializationError:
            return "hipErrorInitializationError";

        case hipErrorLaunchFailure:
            return "hipErrorLaunchFailure";

        case hipErrorPriorLaunchFailure:
            return "hipErrorPriorLaunchFailure";

        case hipErrorLaunchOutOfResources:
            return "hipErrorLaunchOutOfResources";

        case hipErrorInvalidDeviceFunction:
            return "hipErrorInvalidDeviceFunction";

        case hipErrorInvalidConfiguration:
            return "hipErrorInvalidConfiguration";

        case hipErrorInvalidDevice:
            return "hipErrorInvalidDevice";

        case hipErrorInvalidValue:
            return "hipErrorInvalidValue";

        case hipErrorInvalidSymbol:
            return "hipErrorInvalidSymbol";

        case hipErrorMapBufferObjectFailed:
            return "hipErrorMapBufferObjectFailed";

        case hipErrorInvalidDevicePointer:
            return "hipErrorInvalidDevicePointer";

        case hipErrorInvalidMemcpyDirection:
            return "hipErrorInvalidMemcpyDirection";

        case hipErrorUnknown:
            return "hipErrorUnknown";

        case hipErrorInvalidResourceHandle:
            return "hipErrorInvalidResourceHandle";

        case hipErrorNotReady:
            return "hipErrorNotReady";

        case hipErrorInsufficientDriver:
            return "hipErrorInsufficientDriver";

        case hipErrorSetOnActiveProcess:
            return "hipErrorSetOnActiveProcess";

        case hipErrorNoDevice:
            return "hipErrorNoDevice";

        case hipErrorECCNotCorrectable:
            return "hipErrorECCNotCorrectable";

        case hipErrorSharedObjectSymbolNotFound:
            return "hipErrorSharedObjectSymbolNotFound";

        case hipErrorSharedObjectInitFailed:
            return "hipErrorSharedObjectInitFailed";

        case hipErrorUnsupportedLimit:
            return "hipErrorUnsupportedLimit";

        case hipErrorPeerAccessAlreadyEnabled:
            return "hipErrorPeerAccessAlreadyEnabled";

        case hipErrorPeerAccessNotEnabled:
            return "hipErrorPeerAccessNotEnabled";

        case hipErrorProfilerDisabled:
            return "hipErrorProfilerDisabled";

        case hipErrorProfilerNotInitialized:
            return "hipErrorProfilerNotInitialized";

        case hipErrorProfilerAlreadyStarted:
            return "hipErrorProfilerAlreadyStarted";

        case hipErrorProfilerAlreadyStopped:
            return "hipErrorProfilerAlreadyStopped";

        case hipErrorAssert:
            return "hipErrorAssert";

        case hipErrorHostMemoryAlreadyRegistered:
            return "hipErrorHostMemoryAlreadyRegistered";

        case hipErrorHostMemoryNotRegistered:
            return "hipErrorHostMemoryNotRegistered";

        case hipErrorOperatingSystem:
            return "hipErrorOperatingSystem";

        case hipErrorPeerAccessUnsupported:
            return "hipErrorPeerAccessUnsupported";

        case hipErrorIllegalAddress:
            return "hipErrorIllegalAddress";

        case hipErrorInvalidGraphicsContext:
            return "hipErrorInvalidGraphicsContext";

        default:
            return "<unknown>";
    }

    return "<unknown>";
}

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET hipDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "HIP error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _hipGetErrorEnum(result), func);
        DEVICE_RESET
        // Make sure we call HIP Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

// This will output the proper HIP error strings in the event that a HIP host call returns an error
#define checkHipErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling hipGetLastError
#define getLastHipError(msg)      __getLastHipError (msg, __FILE__, __LINE__)

inline void __getLastHipError(const char *errorMessage, const char *file, const int line)
{
    hipError_t err = hipGetLastError();

    if (hipSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastHipError() HIP error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, hipGetErrorString(err));
        DEVICE_RESET
        exit(EXIT_FAILURE);
    }
}

// This will only print the proper error string when calling hipGetLastError but not exit program incase error detected.
#define printLastHipError(msg)      __printLastHipError (msg, __FILE__, __LINE__)

inline void __printLastHipError(const char *errorMessage, const char *file, const int line)
{
    hipError_t err = hipGetLastError();

    if (hipSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastHipError() HIP error : %s : (%d) %s.\n",
            file, line, errorMessage, (int)err, hipGetErrorString(err));
    }
}


#endif
