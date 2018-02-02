package touchstone.ai.panorama;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;


public class MenuActivity extends AppCompatActivity {

    private static final String TAG = "MenuActivity";

    static {
        if(!OpenCVLoader.initDebug()){
            Log.d(TAG, "OpenCV not loaded");
        } else {
            Log.d(TAG, "OpenCV loaded");
        }
    }

    private static int PICK_IMAGE_REQUEST = 1;
    Uri imageUri;
    ImageView imageView;
    Bitmap src_bitmap, bitmap;
    Mat src_mat,image_mat;
    InputStream istr;
    Spinner cameraList;
    String cameraMode;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);
        imageView = findViewById(R.id.imageView);
        AssetManager assetManager = getAssets();
        try{
            istr = assetManager.open("a.jpg");
        }
        catch (IOException e){
            e.printStackTrace();
        }
        Bitmap background = BitmapFactory.decodeStream(istr);
        imageView.setImageBitmap(background);
        // Spinner element
        cameraList = (Spinner) findViewById(R.id.cameraList);
        List<String> list = new ArrayList<>();
        list.add("Linear panorama");
        list.add("Cabinet detection");
        ArrayAdapter<String> adapter = new ArrayAdapter(this, android.R.layout.simple_spinner_item,list);
        adapter.setDropDownViewResource(android.R.layout.simple_list_item_single_choice);
        cameraList.setAdapter(adapter);
        cameraList.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                cameraMode =  cameraList.getSelectedItem().toString();
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
            }
        });
    }


    public void openClicked(View v){
        Intent intent = new Intent();
        // Show only images, no videos or anything else
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        // Always show the chooser (if there are multiple options available)
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE_REQUEST){
            imageUri = data.getData();
            try{
                src_bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
            }
            catch (IOException e) {
                e.printStackTrace();
            }
            imageView.setImageBitmap(src_bitmap);
        }
    }

    public void detectClicked(View v){
        bitmap = src_bitmap.copy(src_bitmap.getConfig(), true);
        src_mat = new Mat(bitmap.getHeight(),bitmap.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(bitmap,src_mat);
        image_mat = src_mat.clone();
        //canny edge
        Imgproc.cvtColor(image_mat,image_mat,Imgproc.COLOR_RGB2GRAY);
        Imgproc.Canny(image_mat,image_mat,30,200);
        ////close edge
        Mat kernel = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(5,5));
        Imgproc.morphologyEx(image_mat,image_mat,Imgproc.MORPH_CLOSE,kernel);
        //get contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierachy = new Mat();
        Imgproc.findContours(image_mat,contours,hierachy,Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);
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
            long image_size = image_mat.height()*image_mat.width();
            //if estimate shape has 4 points and this shape area>1/72 image size
            if (approxCurve.total() == 4 && Imgproc.contourArea(points)>= image_size/300 &&  Imgproc.contourArea(points)<= image_size/3){
                List<MatOfPoint> cnt = new ArrayList<MatOfPoint>();
                //create list to add points to draw
                cnt.add(points);

                Moments M = Imgproc.moments(points);
                if (M.m00 != 0) {
                    int cx = (int)M.m10/(int)M.m00;
                    int cy = (int)M.m01/(int)M.m00;
                    Point center = new Point(cx,cy);
                    Imgproc.drawMarker(src_mat,center,new Scalar(0,255,0),Imgproc.MARKER_STAR,2,4,1);
                    Imgproc.drawContours(src_mat,cnt,-1,new Scalar(0,0,255),2);
                    Point A = points.toList().get(0);
                    Point B = points.toList().get(1);
                    Point C = points.toList().get(2);
                    Point D = points.toList().get(3);
                    long AB = (long)Math.hypot(A.x-B.x, A.y-B.y);
                    long BC = (long)Math.hypot(B.x-C.x, B.y-C.y);
                    long CD = (long)Math.hypot(C.x-D.x, C.y-D.y);
                    long DA = (long)Math.hypot(D.x-A.x, D.y-A.y);
                    Imgproc.putText(src_mat,""+AB, midPoint(A,B),1,1, new Scalar (0,255,0),1);
                    Imgproc.putText(src_mat,""+BC, midPoint(B,C),1,1, new Scalar (0,255,0),1);
                    Imgproc.putText(src_mat,""+CD, midPoint(C,D),1,1, new Scalar (0,255,0),1);
                    Imgproc.putText(src_mat,""+DA, midPoint(D,A),1,1, new Scalar (0,255,0),1);
                }
            }
        }
        Utils.matToBitmap(src_mat,bitmap);
        imageView.setImageBitmap(bitmap);
    }






    Point midPoint(Point A, Point B){
        Point Mid = new Point();
        Mid.x = (A.x+B.x)/2;
        Mid.y = (A.y+B.y)/2;
        return Mid;
    }




}





