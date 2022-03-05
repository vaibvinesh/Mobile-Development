package edu.asu.ame.meteor.speedytiltshift2022;

import android.graphics.Bitmap;
import android.os.SystemClock;

public class SpeedyTiltShift {
    private static float runTime = 0.0f;
    static SpeedyTiltShift Singleton = new SpeedyTiltShift();

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }


    public static Bitmap tiltshift_java(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        //cannot write to input Bitmap, since it may be immutable
        //if you try, you may get a java.lang.IllegalStateException

        /** Benchmark timer start
         *  */
        float rtCheck1 = SystemClock.elapsedRealtime();
        // replacing the statements here
        // in order to operator the tilt-shift bitmap

        /** *********************************************************************
         *  bitmap -> array
         * **********************************************************************/
        int size = input.getHeight()*input.getWidth();
        int[] pixels = new int[size];
        int[] pixelsOut = new int[size];

        double[] blue = new double[size];
        double[] green = new double[size];
        double[] red = new double[size];

        /* store pixels to the int array */
        // 32 bits per pixel
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());

        for (int i=0; i<size; i++){
            int B = pixels[i]%0x100;        // pBlue = (p & 0xff) == (p % 0x100)
            int G = (pixels[i]>>8)%0x100;   // shift 8 bits
            int R = (pixels[i]>>16)%0x100;  // shift 16 bits
            int A = 0xff;                   // how transparent the pixel is or not  // 0xff not transparent at all

            // G=0; //Temporary: set all green values to 0 as a demo.
            blue[i] = B;
            green[i] = G;
            red[i] = R;
        }
        /** *********************************************************************
         *  Gaussian Blur filter
         *  size of kernel = [2r+1]x[2r+1], where r = Math.ceil(2*sigma)
         *  then,
         *      x = [-r, r]
         *      u = [-r, r]
         *  Blur(int[] pixelsOut, int currIndex, int width, float sigma)
         * **********************************************************************/
        double[] qBlue = new double[size];
        double[] qGreen = new double[size];
        double[] qRed = new double[size];

        boolean vertical = false;
        for (int i=0; i < size; i++) {
            /* Far Gaussian blur */
            if (i < (a0+1)*input.getWidth()) {  // height: 0~a0, width: input.getWidth()
                qBlue[i] = farBlur(blue, i, input.getWidth(), input.getHeight(), sigma_far, vertical);   // width * (a0+1)
                qGreen[i] = farBlur(green, i, input.getWidth(), input.getHeight(), sigma_far, vertical);
                qRed[i] = farBlur(red, i, input.getWidth(), input.getHeight(), sigma_far, vertical);
            }
            /* Gradient Far Gaussian blur */
            else if ((i < (a1+1)*input.getWidth()) && (i >= ((a0+1)*input.getWidth()) )) {
                qBlue[i] = gFarBlur(blue, i, input.getWidth(), input.getHeight(), sigma_far, a1, a0, vertical);
                qGreen[i] = gFarBlur(green, i, input.getWidth(), input.getHeight(), sigma_far, a1, a0, vertical);
                qRed[i] = gFarBlur(red, i, input.getWidth(), input.getHeight(), sigma_far, a1, a0, vertical);
            }
            /* Focused */
            else if ((i < (a2+1)*input.getWidth()) && (i >= ((a1+1)*input.getWidth()))) {
                qBlue[i] = blue[i];
                qGreen[i] = green[i];
                qRed[i] = red[i];
            }
            /* Gradient Near Gaussian blur */
            else if ((i < (a3+1)*input.getWidth()) && (i >= ((a2+1)*input.getWidth()))) {
                qBlue[i] = gNearBlur(blue, i, input.getWidth(), input.getHeight(), sigma_near, a3, a2, vertical);
                qGreen[i] = gNearBlur(green, i, input.getWidth(), input.getHeight(), sigma_near, a3, a2, vertical);
                qRed[i] = gNearBlur(red, i, input.getWidth(), input.getHeight(), sigma_near, a3, a2, vertical);
            }
            /* Near Gaussian blur */
            else {
                qBlue[i] = nearBlur(blue, i, input.getWidth(), input.getHeight(), sigma_near, vertical);
                qGreen[i] = nearBlur(green, i, input.getWidth(), input.getHeight(), sigma_near, vertical);
                qRed[i] = nearBlur(red, i, input.getWidth(), input.getHeight(), sigma_near, vertical);
            }
        }

        /* Vertical Conv */
        vertical = true;
        for (int i=0; i < size; i++) {
            double B = 0;
            double G = 0;
            double R = 0;
            int A = 0xff;                   // how transparent the pixel is or not  // 0xff not transparent at all
            /* Far Gaussian blur */
            if (i < (a0+1)*input.getWidth()) {  // height: 0~a0, width: input.getWidth()
                B = farBlur(qBlue, i, input.getWidth(), input.getHeight(), sigma_far, vertical);   // width * (a0+1)
                G = farBlur(qGreen, i, input.getWidth(), input.getHeight(), sigma_far, vertical);
                R = farBlur(qRed, i, input.getWidth(), input.getHeight(), sigma_far, vertical);
            }
            /* Gradient Far Gaussian blur */
            else if ((i < (a1+1)*input.getWidth()) && (i >= ((a0+1)*input.getWidth()) )) {
                B = gFarBlur(qBlue, i, input.getWidth(), input.getHeight(), sigma_far, a1, a0, vertical);
                G = gFarBlur(qGreen, i, input.getWidth(), input.getHeight(), sigma_far, a1, a0, vertical);
                R = gFarBlur(qRed, i, input.getWidth(), input.getHeight(), sigma_far, a1, a0, vertical);
            }
            /* Focused */
            else if ((i < (a2+1)*input.getWidth()) && (i >= ((a1+1)*input.getWidth()))) {
                B = (int)blue[i];    // no changes from horizontal direction
                G = (int)green[i];
                R = (int)red[i];
            }
            /* Gradient Near Gaussian blur */
            else if ((i < (a3+1)*input.getWidth()) && (i >= ((a2+1)*input.getWidth()))) {
                B = gNearBlur(qBlue, i, input.getWidth(), input.getHeight(), sigma_near, a3, a2, vertical);
                G = gNearBlur(qGreen, i, input.getWidth(), input.getHeight(), sigma_near, a3, a2, vertical);
                R = gNearBlur(qRed, i, input.getWidth(), input.getHeight(), sigma_near, a3, a2, vertical);
            }
            /* Near Gaussian blur */
            else {
                B = nearBlur(qBlue, i, input.getWidth(), input.getHeight(), sigma_near, vertical);
                G = nearBlur(qGreen, i, input.getWidth(), input.getHeight(), sigma_near, vertical);
                R = nearBlur(qRed, i, input.getWidth(), input.getHeight(), sigma_near, vertical);
            }

            pixelsOut[i] = (A & 0xff) << 24 | ((int)Math.ceil(R) & 0xff) << 16 | ((int)Math.ceil(G) & 0xff) << 8 | ((int)Math.ceil(B) & 0xff);
        }
        /***********************************************************************
         * output
         * **********************************************************************/
        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());

        /** Benchmark timer end
         * */
        runTime = SystemClock.elapsedRealtime()- rtCheck1; // benchmark end

        return outBmp;
    }
    public static Bitmap tiltshift_cpp(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        float rtCheck1 = SystemClock.elapsedRealtime();
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());

        tiltshiftcppnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);

        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        runTime = SystemClock.elapsedRealtime()- rtCheck1;
        return outBmp;
    }
    public static Bitmap tiltshift_neon(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        float rtCheck1 = SystemClock.elapsedRealtime();
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());

        tiltshiftneonnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);

        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        runTime = SystemClock.elapsedRealtime()- rtCheck1;
        return outBmp;
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public static native int tiltshiftcppnative(int[] inputPixels, int[] outputPixels, int width, int height, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3);
    public static native int tiltshiftneonnative(int[] inputPixels, int[] outputPixels, int width, int height, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3);

    /***********************************************************************
     * Gaussian Blur methods implementation
     * Blur(int[] pixelsOut, int currIndex, int width)
     * **********************************************************************/

    private static void kernel(double[] blurKernel,int r, float sigma) {
        double denom = Math.sqrt(1.0/(2*Math.PI*sigma*sigma));
        for (int k=0; k < 2*r+1; k++) {
            blurKernel[k] = ( denom * Math.exp((-(k-r)*(k-r))/(2.0*sigma*sigma) ));
        }
    }

    //Fast Convolution
    private static double conv(double[] pixels, int index,int width, int height, double[] blurKernel, int r, boolean vertical) {
        int x = index % width;
        int y = index / width;
        double temp = 0;

        /* choose for vertical or horizontal case */
        for (int j = -r; j <= r; j++) {
            if (vertical) {
                if (y + j < 0 || y + j >= height) {  // edge cases
                    temp += blurKernel[j + r] * pixels[(y - j) * width + (x)];   // symmetric padding
                } else {
                    temp += blurKernel[j + r] * pixels[(y + j) * width + (x)];
                }
            }
            else {
                if (x + j < 0 || x + j >= width) {  // edge cases
                    temp += blurKernel[j + r] * pixels[y * width + (x - j)];   // symmetric padding
                } else {
                    temp += blurKernel[j + r] * pixels[y * width + (x + j)];
                }
            }

        }
        return temp;
    }

    // Far Blur
    private static double farBlur(double[] pixels, int currIndex, int width, int height, float sigma, boolean vertical) {
        int r = (int) Math.ceil(2*sigma);

        /* Define far blurred filter */
        double[] blurKernel = new double[2*r+1];
        kernel(blurKernel, r, sigma);


        /* Convolution */
        double pixel = pixels[currIndex];
        if (sigma > 0.6){
            pixel = conv(pixels, currIndex, width, height, blurKernel, r, vertical);
        }
        return pixel;
    }

    //Gradient blur for far.
    private static double gFarBlur(double[] pixels, int currIndex, int width, int height, float sigma, int a1, int a0, boolean vertical) {
        /* Gradient sigma calculation */
        float gSigma = sigma * ((a1 - ((float) currIndex) / width)/(a1 - a0)); // height = a1 - a0

        int r = (int) Math.ceil(2*gSigma);

        /* Define far blurred filter */
        double[] blurKernel = new double[2*r+1];
        kernel(blurKernel, r, gSigma);

        /* Convolution */
        double pixel = pixels[currIndex];
        if (gSigma > 0.6){
            pixel = conv(pixels, currIndex, width, height, blurKernel, r, vertical);
        }
        return pixel;
    }

    // Gradient Near blur
    private static double gNearBlur(double[] pixels, int currIndex, int width, int height, float sigma, int a3, int a2, boolean vertical) {
        /* Gradient sigma calculation */
        float gSigma = sigma * ((((float) currIndex) / width - a2)/(a3 - a2)); // height = a3 - a2

        int r = (int) Math.ceil(2*gSigma);

        /* Define far blurred filter */
        double[] blurKernel = new double[2*r+1];
        kernel(blurKernel, r, gSigma);

        /* Convolution */
        double pixel = pixels[currIndex];
        if (gSigma > 0.6){
            pixel = conv(pixels, currIndex, width, height, blurKernel, r, vertical);
        }
        return pixel;
    }

    // Near Blur
    private static double nearBlur(double[] pixels, int currIndex, int width, int height, float sigma, boolean vertical) {
        int r = (int) Math.ceil(2*sigma);

        /* Define far blurred filter */
        double[] blurKernel = new double[2*r+1];
        kernel(blurKernel, r, sigma);

        /* Convolution */
        double pixel = pixels[currIndex];
        if (sigma > 0.6){
            pixel = conv(pixels, currIndex, width, height, blurKernel, r, vertical);
        }
        return pixel;
    }

    public static float getRunTime(){ //getter
        return runTime;
    }
}