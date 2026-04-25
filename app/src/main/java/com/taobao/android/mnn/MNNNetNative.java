package com.taobao.android.mnn;

import android.graphics.Bitmap;
import android.util.Log;

public class MNNNetNative {
    private static final String TAG = "MNN";
    private static boolean sLibraryLoaded = false;

    static {
        Log.d(TAG, "MNNNetNative static init");
    }

    // load libraries
    static void loadGpuLibrary(String name) {
        try {
            System.loadLibrary(name);
            Log.d(TAG, "Loaded GPU library: " + name);
        } catch (Throwable ce) {
            Log.w(TAG, "load MNN " + name + " GPU so exception: " + ce.getMessage());
        }
    }

    public static synchronized boolean loadLibraries() {
        if (sLibraryLoaded) {
            Log.d(TAG, "MNN libraries already loaded");
            return true;
        }
        Log.d(TAG, "Start loading MNN libraries...");
        try {
            Log.d(TAG, "Loading c++_shared...");
            System.loadLibrary("c++_shared");
            Log.d(TAG, "Loading MNN...");
            System.loadLibrary("MNN");
            Log.d(TAG, "Loading MNN JNI core...");
            System.loadLibrary("mnncore");
            Log.d(TAG, "Loading GPU libraries...");
            loadGpuLibrary("MNN_Vulkan");
            loadGpuLibrary("MNN_CL");
            loadGpuLibrary("MNN_GL");
            sLibraryLoaded = true;
            Log.d(TAG, "MNN libraries loaded successfully");
            return true;
        } catch (Throwable e) {
            Log.e(TAG, "Failed to load MNN libraries: " + e.getMessage(), e);
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
