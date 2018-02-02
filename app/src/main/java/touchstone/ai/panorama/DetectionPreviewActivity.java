package touchstone.ai.panorama;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.w3c.dom.Text;

import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;


public class DetectionPreviewActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2,View.OnTouchListener {

    private static final String TAG = "OCVSample::Activity";
    private CameraBridgeViewBase mOpenCvCameraView;
    Mat img1;
    Mat rgba;
    int w,h;

    static {
        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    try {
                        initializeOpenCVDependencies();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private void initializeOpenCVDependencies() throws IOException {
        mOpenCvCameraView.enableView();
    }


    public DetectionPreviewActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {

        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_cabinet);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.cabinetOpenCvCamera);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setOnTouchListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        w = width;
        h = height;
    }

    public void onCameraViewStopped() {
    }


    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        rgba = inputFrame.rgba();
        Mat image = rgba.clone();
        //process image to gray
        Imgproc.cvtColor(image,image,Imgproc.COLOR_RGB2GRAY);
        //find edge
        Imgproc.Canny(image,image,30,210);
        ////close edge
        Mat kernel = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(5,5));
        Imgproc.morphologyEx(image,image,Imgproc.MORPH_CLOSE,kernel);
        //get contours
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierachy = new Mat();
        Imgproc.findContours(image,contours,hierachy,Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);
        //loop over contours
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        for (int i=0; i<contours.size();i++){
            //Convert contours(i) from MatOfPoint to MatOfPoint2f
            MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(i).toArray());
            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double peri = Imgproc.arcLength(contour2f, true) * 0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, peri,true);
            // Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint(approxCurve.toArray());
            long image_size = rgba.height()*rgba.width();
            //if estimate shape has 4 points and this shape area>1/72 image size
            if (approxCurve.total() == 4 && Imgproc.contourArea(points)>= image_size/300 && Imgproc.contourArea(points) <= image_size/3){
                List<MatOfPoint> cnt = new ArrayList<MatOfPoint>();
                //create list to add points to draw
                cnt.add(points);

                Moments M = Imgproc.moments(points);
                if (M.m00 != 0) {
                    int cx = (int)M.m10/(int)M.m00;
                    int cy = (int)M.m01/(int)M.m00;
                    Point center = new Point(cx,cy);
                    Imgproc.drawMarker(rgba,center,new Scalar(0,255,0),Imgproc.MARKER_STAR,2,4,1);
                    Imgproc.drawContours(rgba,cnt,-1,new Scalar(0,0,255),2);
                    Point A = points.toList().get(0);
                    Point B = points.toList().get(1);
                    Point C = points.toList().get(2);
                    Point D = points.toList().get(3);
                    long AB = (long)Math.hypot(A.x-B.x, A.y-B.y);
                    long BC = (long)Math.hypot(B.x-C.x, B.y-C.y);
                    long CD = (long)Math.hypot(C.x-D.x, C.y-D.y);
                    long DA = (long)Math.hypot(D.x-A.x, D.y-A.y);
                    Imgproc.putText(rgba,""+AB, midPoint(A,B),1,1, new Scalar (0,255,0),1);
                    Imgproc.putText(rgba,""+BC, midPoint(B,C),1,1, new Scalar (0,255,0),1);
                    Imgproc.putText(rgba,""+CD, midPoint(C,D),1,1, new Scalar (0,255,0),1);
                    Imgproc.putText(rgba,""+DA, midPoint(D,A),1,1, new Scalar (0,255,0),1);
                }
            }
        }
        return rgba;
    }

    Point midPoint(Point A, Point B){
        Point Mid = new Point();
        Mid.x = (A.x+B.x)/2;
        Mid.y = (A.y+B.y)/2;
        return Mid;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Toast.makeText(this, "touched", Toast.LENGTH_SHORT).show();
        return false;
        /*if (choice==false) choice = true;
        else choice = false;
        int i = 0;
        if (i < mOpenCvCameraView.getResolutionList().size()) {
            Toast.makeText(this, "" +  mOpenCvCameraView.getResolutionList().get(i).width + "x" + mOpenCvCameraView.getResolutionList().get(i).height, Toast.LENGTH_SHORT).show();
            i++;
        }
        else i=0;
        return false;*/

    }

}
