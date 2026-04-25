package com.taobao.android.mnn;

import android.graphics.Bitmap;

public class MNNNetNative {
    private static boolean sLibraryLoaded = false;

    static void loadGpuLibrary(String name) {
        try {
            System.loadLibrary(name);
        } catch (Throwable ignored) {
        }
    }

    public static synchronized boolean loadLibraries() {
        if (sLibraryLoaded) {
            return true;
        }
        try {
            System.loadLibrary("c++_shared");
            System.loadLibrary("MNN");
            System.loadLibrary("mnncore");
            loadGpuLibrary("MNN_Vulkan");
            loadGpuLibrary("MNN_CL");
            loadGpuLibrary("MNN_GL");
            sLibraryLoaded = true;
            return true;
        } catch (Throwable e) {
            return false;
        }
    }

    //Net
    protected static native long nativeCreateNetFromFile(String modelName);

    protected static native long nativeCreateNetFromBuffer(byte[] buffer);

    protected static native long nativeReleaseNet(long netPtr);


    //Session
    protected static native long nativeCreateSession(long netPtr, int forwardType, int numThread, String[] saveTensors, String[] outputTensors);

    protected static native void nativeReleaseSession(long netPtr, long sessionPtr);

    protected static native int nativeRunSession(long netPtr, long sessionPtr);

    protected static native int nativeRunSessionWithCallback(long netPtr, long sessionPtr, String[] nameArray, long[] tensorAddr);

    protected static native int nativeReshapeSession(long netPtr, long sessionPtr);

    protected static native long nativeGetSessionInput(long netPtr, long sessionPtr, String name);

    protected static native long nativeGetSessionOutput(long netPtr, long sessionPtr, String name);


    //Tensor
    protected static native void nativeReshapeTensor(long netPtr, long tensorPtr, int[] dims);

    protected static native int[] nativeTensorGetDimensions(long tensorPtr);

    protected static native void nativeSetInputIntData(long netPtr, long tensorPtr, int[] data);

    protected static native void nativeSetInputFloatData(long netPtr, long tensorPtr, float[] data);


    //If dest is null, return length
    protected static native int nativeTensorGetData(long tensorPtr, float[] dest);

    protected static native int nativeTensorGetIntData(long tensorPtr, int[] dest);

    protected static native int nativeTensorGetUINT8Data(long tensorPtr, byte[] dest);


    //ImageProcess
    protected static native boolean nativeConvertBitmapToTensor(Bitmap srcBitmap, long tensorPtr, int destFormat, int filterType, int wrap, float[] matrixValue, float[] mean, float[] normal);

    protected static native boolean nativeConvertBufferToTensor(byte[] bufferData, int width, int height, long tensorPtr,
                                                                int srcFormat, int destFormat, int filterType, int wrap, float[] matrixValue, float[] mean, float[] normal);

}
