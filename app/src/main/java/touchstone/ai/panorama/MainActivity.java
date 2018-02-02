package touchstone.ai.panorama;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaPlayer;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
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
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.w3c.dom.Text;

import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;


public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private int w, h;
    private CameraBridgeViewBase mOpenCvCameraView;
    Button captureButton;
    TextView text1;
    FeatureDetector detector;
    DescriptorExtractor descriptor;
    DescriptorMatcher matcher;
    Mat descriptors2,descriptors1;
    LinkedList<DMatch> good_matches;

    Mat img1;
    Mat rgba;
    MatOfKeyPoint keypoints1,keypoints2;
    int flow=0;
    int directionGuide = 0;
    int toRightThres = 2;
    int toLeftThres = 33;

    TextView timerTextView, textView;
    MediaPlayer mp;

    CountDownTimer Timer;
    boolean pendingCapture = false;

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
        detector = FeatureDetector.create(FeatureDetector.ORB);
        descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        mp = MediaPlayer.create(this, R.raw.cam2);
        /*img1 = new Mat();
        AssetManager assetManager = getAssets();
        InputStream istr = assetManager.open("c.jpg");
        Bitmap bitmap = BitmapFactory.decodeStream(istr);
        Utils.bitmapToMat(bitmap, img1);
        Imgproc.cvtColor(img1, img1, Imgproc.COLOR_RGB2GRAY);
        img1.convertTo(img1, 0); //converting the image to match with the type of the cameras image*/
    }


    public MainActivity() {

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
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.OpenCvCamera);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        captureButton = findViewById(R.id.captureButton);
        textView = findViewById(R.id.textView);
        timerTextView = findViewById(R.id.timerTextView);
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

    public Mat recognize(Mat aInputFrame) {
        //if there is 1 captured then flow>0
        if (flow>0){
            descriptors2 = new Mat();
            keypoints2 = new MatOfKeyPoint();
            detector.detect(aInputFrame, keypoints2);
            descriptor.compute(aInputFrame, keypoints2, descriptors2);

            // Matching
            MatOfDMatch matches = new MatOfDMatch();
            matcher.match(descriptors1, descriptors2, matches);

            List<DMatch> matchesList = matches.toList();
            Double max_dist = 0.0;
            Double min_dist = 100.0;

            //adjust max,min distance base on match list
            good_matches = new LinkedList<DMatch>();

            for (int i = 0; i < matchesList.size(); i++) {
                Double dist = (double) matchesList.get(i).distance;
                if (dist < min_dist)
                    min_dist = dist;
                if (dist > max_dist)
                    max_dist = dist;
            }

            //only get the mathces that have distance <=2 min_dist
            for (int i = 0; i < matchesList.size(); i++) {
                if (matchesList.get(i).distance <= (1.5 * min_dist)){
                    if (matchesList.get(i).distance == min_dist){
                        good_matches.addFirst(matchesList.get(i));
                    }
                    else good_matches.addLast(matchesList.get(i));
                }
            }


            //convert good matches to MatofMatch
            MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(good_matches);
            MatOfByte drawnMatches = new MatOfByte();
            if (aInputFrame.empty() || aInputFrame.cols() < 1 || aInputFrame.rows() < 1) {
                return aInputFrame;
            }
            //number of matches require to draw match
            if (goodMatches.toList().size() > 0){
                int[] vote = new int[37]; //number of matches that have angle of i (0-360)
                int[] angle = new int[goodMatches.toList().size()]; //angle of match i
                for (int i=0; i<goodMatches.toList().size(); i++){
                    //get A and B coordinate
                    Point A = keypoints1.toList().get(goodMatches.toList().get(i).queryIdx).pt; //current frame keypoint
                    Point B = keypoints2.toList().get(goodMatches.toList().get(i).trainIdx).pt; //last picture keypoint
                    //increase vote of this angle
                    vote[(int)getAngle(A,B)/10]++;
                    //store angle of this match
                    angle[i] = (int)getAngle(A,B);
                }
                //find the max vote => the highest frequency angle => true angle
                int maxFrequency = 0;
                int maxAngle = 0;
                for (int i=0; i<vote.length; i++){
                    if (vote[i]>maxFrequency) {
                        maxFrequency = vote[i];
                        maxAngle = i;
                    }
                }
                //auto capture if true angle = threshold left,right
                autoSnap(maxAngle,aInputFrame);


                //debug text
                switch(directionGuide){
                    case 0:
                        Imgproc.putText(aInputFrame,"MOVE LEFT OR RIGHT",new Point(100,100),2,1,new Scalar(255,0,0));
                        break;
                    case 1: //right
                        if (maxAngle<36 && maxAngle>toLeftThres) Imgproc.putText(aInputFrame,"MOVE BACK",new Point(100,100),2,1,new Scalar(255,0,0));
                        if (maxAngle>0 && maxAngle<toRightThres) Imgproc.putText(aInputFrame,"KEEP MOVING",new Point(100,100),2,1,new Scalar(255,0,0));
                        break;
                    case -1: //left
                        if (maxAngle<36 && maxAngle>toLeftThres) Imgproc.putText(aInputFrame,"KEEP MOVING",new Point(100,100),2,1,new Scalar(255,0,0));
                        if (maxAngle>0 && maxAngle<toRightThres) Imgproc.putText(aInputFrame,"MOVE BACK",new Point(100,100),2,1,new Scalar(255,0,0));
                        break;
                }
                Imgproc.putText(aInputFrame,"current: "+angle[0] +"(thres:15)",new Point(100,200),2,1,new Scalar(255,0,0));

                return aInputFrame;
            }
            return aInputFrame;
        }
        return aInputFrame;
    }


    public double getAngle(Point A, Point B) {
        //reproject A and B as we combine and resize them into outputImg, x and col are height
        A.x = A.x/2;
        B.x = B.x/2 + rgba.cols()/2;
        double angle = Math.round(Math.toDegrees(Math.atan2( B.y - A.y,B.x - A.x)));
        if (angle<0) return (angle+360);
        else return angle;
    }

    public boolean autoSnap(int voteMaxAngle, Mat aInputFrame){
        if ((voteMaxAngle==toRightThres && directionGuide==0)||(voteMaxAngle==toRightThres && directionGuide==1) && !pendingCapture) {
            pendingCapture = true;
            directionGuide = 1; //right
            holdPosition(aInputFrame);
            return true;
        }
        if ((voteMaxAngle==toLeftThres && directionGuide==0)||(voteMaxAngle==toLeftThres && directionGuide==-1) && !pendingCapture) {
            pendingCapture = true;
            directionGuide = -1; //left
            holdPosition(aInputFrame);
            return true;
        }
        return false;
    }

    public void saveImage(){
        //clone and convert to gray
        img1 = rgba.clone();
        Imgproc.cvtColor(img1,img1,Imgproc.COLOR_BGR2GRAY);
        //get kp and ds of captured image
        descriptors1 = new Mat();
        keypoints1 = new MatOfKeyPoint();
        detector.detect(img1, keypoints1);
        descriptor.compute(img1, keypoints1, descriptors1);

        //rotate image to save
        Core.transpose(rgba,rgba);
        Core.flip(rgba,rgba,+1);

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        String fileName = Environment.getExternalStorageDirectory().getPath() + "/DCIM/Camera/" + currentDateandTime + ".jpg";
        Imgproc.cvtColor(rgba,rgba,Imgproc.COLOR_RGBA2BGR);
        Imgcodecs.imwrite(fileName,rgba);
        scanFile(fileName); //make it appears in gallery
    }

    private void scanFile(String path) {
        MediaScannerConnection.scanFile(MainActivity.this,
                new String[] { path }, null,
                new MediaScannerConnection.OnScanCompletedListener() {

                    public void onScanCompleted(String path, Uri uri) {
                        Log.i("TAG", "Finished scanning " + path);
                    }
                });
    }

    public void captureClicked(View v){
        if (flow==0){
            //play capture sound
            mp.start();
            captureButton.setText("STOP");
            //clone and convert to gray
            img1 = rgba.clone();
            Imgproc.cvtColor(img1,img1,Imgproc.COLOR_BGR2GRAY);
            //get kp and ds of captured image
            descriptors1 = new Mat();
            keypoints1 = new MatOfKeyPoint();
            detector.detect(img1, keypoints1);
            descriptor.compute(img1, keypoints1, descriptors1);

            flow++;
            textView.setText(String.valueOf(flow));

            //rotate image to save
            Core.transpose(rgba,rgba);
            Core.flip(rgba,rgba,+1);

            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
            String currentDateandTime = sdf.format(new Date());
            String fileName = Environment.getExternalStorageDirectory().getPath() + "/DCIM/Camera/" + currentDateandTime + ".jpg";
            Imgproc.cvtColor(rgba,rgba,Imgproc.COLOR_RGBA2BGR);
            Imgcodecs.imwrite(fileName,rgba);
            Toast.makeText(this, fileName + " saved", Toast.LENGTH_SHORT).show();
            scanFile(fileName);
        }
        else{
            flow=0;
            directionGuide=0;
            textView.setText(String.valueOf(flow));
            captureButton.setText("START");
        }
    }

    public void holdPosition(final Mat aInputFrame){
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                Timer = new CountDownTimer(3000,1000) {
                    @Override
                    public void onTick(long millisUntilFinished) {
                        timerTextView.setText(""+millisUntilFinished/1000);
                    }
                    @Override
                    public void onFinish() {
                        //play capture sound
                        mp.start();
                        saveImage();
                        pendingCapture = false;
                        timerTextView.setText("");
                    }
                }.start();
            }
        });
    }


    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        rgba = inputFrame.rgba();
        return recognize(inputFrame.gray());
    }
}
