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

        int[] blue = new int[size];
        int[] green = new int[size];
        int[] red = new int[size];

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
        for (int i=0; i < size; i++) {
            int color = 0;
            /* Far Gaussian blur */
            if (i < (a0+1)*input.getWidth()) {  // height: 0~a0, width: input.getWidth()
                color = farBlur(blue, green, red, i, input.getWidth(), input.getHeight(), sigma_far);   // width * (a0+1)
            }
            /* Gradient Far Gaussian blur */
            else if ((i < (a1+1)*input.getWidth()) && (i >= ((a0+1)*input.getWidth()) )) {
                color = gFarBlur(blue, green, red, i, input.getWidth(), input.getHeight(), sigma_far, a1, a0);
            }
            /* Focused */
            else if ((i < (a2+1)*input.getWidth()) && (i >= ((a1+1)*input.getWidth()))) {
                color = (0xff) << 24 | (red[i] & 0xff) << 16 | (green[i] & 0xff) << 8 | (blue[i] & 0xff);
            }
            /* Gradient Near Gaussian blur */
            else if ((i < (a3+1)*input.getWidth()) && (i >= ((a2+1)*input.getWidth()))) {
                color = gNearBlur(blue, green, red, i, input.getWidth(), input.getHeight(), sigma_near, a3, a2);
            }
            /* Near Gaussian blur */
            else {
                color = nearBlur(blue, green, red, i, input.getWidth(), input.getHeight(), sigma_near);
            }

            pixelsOut[i]=color;
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
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());

        tiltshiftcppnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);

        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        return outBmp;
    }
    public static Bitmap tiltshift_neon(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());

        tiltshiftneonnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);

        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
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

    private static void kernel(double[][] blurKernel,int r, float sigma) {
        double denom = (1.0/(2*Math.PI*sigma*sigma));
        for (int i=0; i < 2*r+1; i++) {
            for (int j=0; j < 2*r+1; j++) {
                blurKernel[i][j] = ( denom * Math.exp((-(i-r)*(i-r)-(j-r)*(j-r))/(2.0*sigma*sigma) ));
            }
        }
    }

    private static int conv(int[] pixels, int index,int width, int height, double[][] blurKernel, int r) {
        int x = index % width;
        int y = index / width;
        float temp = 0;
        for (int i=-r; i <= r; i++) {
            for (int j=-r; j <= r; j++) {
                if ((y+j < 0 && x+i < 0) || (y+j >= height && x+i >= width) || (y+j < 0 && x+i >= width) || (y+j >= height && x+i < 0)) {  // top-left, bottom-right, bottom-left, and top-right corners
                    temp += blurKernel[j+r][i+r] * pixels[(y-j)*width+(x-i)];   // symmetric padding
                }
                else if (y+j < 0 || y+j >= height) {
                    temp += blurKernel[j+r][i+r] * pixels[(y-j)*width+(x+i)];   // symmetric padding
                }
                else if (x+i < 0 || x+i >= width) {
                    temp += blurKernel[j+r][i+r] * pixels[(y+j)*width+(x-i)];   // symmetric padding
                }
                else {
                    temp += blurKernel[j+r][i+r] * pixels[(y+j)*width+(x+i)];
                }
            }
        }
        //System.out.println(Math.ceil(temp));
        return (int) Math.ceil(temp);
    }

    private static int farBlur(int[] blue, int[] green, int[] red, int currIndex, int width, int height, float sigma) {
        int r = (int) Math.ceil(4*sigma);

        /* Define far blurred filter */
        double[][] blurKernel = new double[2*r+1][2*r+1];
        kernel(blurKernel, r, sigma);

        /* Convolution */
        int B = blue[currIndex];
        int G = green[currIndex];
        int R = red[currIndex];
        int A = 0xff;
        if (sigma > 0.6){
            B = conv(blue, currIndex, width, height, blurKernel, r);
            G = conv(green, currIndex, width, height, blurKernel, r);
            R = conv(red, currIndex, width, height, blurKernel, r);
            A = 0xff;                   // how transparent the pixel is or not  // 0xff not transparent at all
        }
        int color = (A & 0xff) << 24 | (R & 0xff) << 16 | (G & 0xff) << 8 | (B & 0xff);
        return color;
    }

    private static int gFarBlur(int[] blue, int[] green, int[] red, int currIndex, int width, int height, float sigma, int a1, int a0) {
        /* Gradient sigma calculation */
        float gSigma = sigma * ((a1 - ((float) currIndex) / width)/(a1 - a0)); // height = a1 - a0

        int r = (int) Math.ceil(4*gSigma);

        /* Define far blurred filter */
        double[][] blurKernel = new double[2*r+1][2*r+1];
        kernel(blurKernel, r, gSigma);

        /* Convolution */
        int B = blue[currIndex];
        int G = green[currIndex];
        int R = red[currIndex];
        int A = 0xff;
        if (gSigma > 0.6){
            B = conv(blue, currIndex, width, height, blurKernel, r);
            G = conv(green, currIndex, width, height, blurKernel, r);
            R = conv(red, currIndex, width, height, blurKernel, r);
            A = 0xff;                   // how transparent the pixel is or not  // 0xff not transparent at all
        }

        int color = (A & 0xff) << 24 | (R & 0xff) << 16 | (G & 0xff) << 8 | (B & 0xff);
        return color;
    }

    private static int gNearBlur(int[] blue, int[] green, int[] red, int currIndex, int width, int height, float sigma, int a3, int a2) {
        /* Gradient sigma calculation */
        float gSigma = sigma * ((((float) currIndex) / width - a2)/(a3 - a2)); // height = a3 - a2

        int r = (int) Math.ceil(4*gSigma);

        /* Define far blurred filter */
        double[][] blurKernel = new double[2*r+1][2*r+1];
        kernel(blurKernel, r, gSigma);

        /* Convolution */
        int B = blue[currIndex];
        int G = green[currIndex];
        int R = red[currIndex];
        int A = 0xff;
        if (gSigma > 0.6){
            B = conv(blue, currIndex, width, height, blurKernel, r);
            G = conv(green, currIndex, width, height, blurKernel, r);
            R = conv(red, currIndex, width, height, blurKernel, r);
            A = 0xff;                   // how transparent the pixel is or not  // 0xff not transparent at all
        }


        int color = (A & 0xff) << 24 | (R & 0xff) << 16 | (G & 0xff) << 8 | (B & 0xff);
        return color;
    }

    private static int nearBlur(int[] blue, int[] green, int[] red, int currIndex, int width, int height, float sigma) {
        int r = (int) Math.ceil(4*sigma);

        /* Define far blurred filter */
        double[][] blurKernel = new double[2*r+1][2*r+1];
        kernel(blurKernel, r, sigma);

        /* Convolution */
        int B = blue[currIndex];
        int G = green[currIndex];
        int R = red[currIndex];
        int A = 0xff;
        if (sigma > 0.6){
            B = conv(blue, currIndex, width, height, blurKernel, r);
            G = conv(green, currIndex, width, height, blurKernel, r);
            R = conv(red, currIndex, width, height, blurKernel, r);
            A = 0xff;                   // how transparent the pixel is or not  // 0xff not transparent at all
        }
        int color = (A & 0xff) << 24 | (R & 0xff) << 16 | (G & 0xff) << 8 | (B & 0xff);
        return color;
    }

    public static float getRunTime(){ //getter
        return runTime;
    }
}
