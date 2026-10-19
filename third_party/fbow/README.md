FBOW inference subset from https://github.com/rmsalinas/fbow at commit
e148dbcbd8ee1976b1b07988b5761a9171b9f01a. See LICENSE.

fbow.cpp and cpu.h are upstream sources. fbow.h adds FBOW_ORB_ONLY to the
upstream guards for unused SIMD code. fbow_exports.h supplies the
build-generated export macro for our private build. Only binary ORB inference
is enabled; the unused floating-point SIMD paths are disabled with
FBOW_ORB_ONLY so the rest of myslam retains its existing compiler/CPU settings.
Vocabulary training and other upstream utilities are not included.
