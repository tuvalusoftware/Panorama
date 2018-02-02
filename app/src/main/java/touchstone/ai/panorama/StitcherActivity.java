package touchstone.ai.panorama;


import android.content.ClipData;
import android.content.Intent;
import android.graphics.Bitmap;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;


public class StitcherActivity extends AppCompatActivity{

    ClipData imageData;
    ArrayList<Uri> imagesList = new ArrayList<>();
    ImageView panoView;
    Mat img_matches;
    Bitmap panoBitmap;
    int direction = -1; //1 = left to right, -1 = right to left
    TextView directionText;
    Switch directionSwitch;
    Spinner detectorList, descriptorList, ransacThresList;
    String Detector, Descriptor, RANSACvalue;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_stitcher);
        panoView = findViewById(R.id.panoView);
        directionText = findViewById(R.id.stitchDirectionText);
        directionSwitch = findViewById(R.id.directionSwitch);
        directionSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    directionText.setText("Stitch from left to right");
                    direction = 1;
                    Collections.sort(imagesList,Collections.<Uri>reverseOrder());
                    System.out.println(Arrays.toString(imagesList.toArray()));
                }
                else {
                    directionText.setText("Stitch from right to left");
                    direction = -1;
                    Collections.sort(imagesList);
                    System.out.println(Arrays.toString(imagesList.toArray()));
                }
            }

        });
        initFeatureLists();
    }

    public void imageClicked(View v) {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), 5);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
        //re-init to empty list
        imagesList = new ArrayList<>();
        if (resultCode == RESULT_OK) {
            if (data.getClipData() != null) {
                imageData = data.getClipData();
                for (int i=0; i<imageData.getItemCount(); i++) {
                    Uri uri = imageData.getItemAt(i).getUri();
                    System.out.println(uri);
                    imagesList.add(uri);
                }
            }
            //sort base on capture direction
            //default is left to right
            if (direction==1){
                Collections.sort(imagesList,Collections.<Uri>reverseOrder());
            }
            else{
                Collections.sort(imagesList);
            }
            //notify
            if (imagesList.size() <= 1 )
                Toast.makeText(this, "You need to choose more than 1 image to stitch", Toast.LENGTH_SHORT).show();
            else
                Toast.makeText(this, "You selected " + imagesList.size() + " images to stitch", Toast.LENGTH_SHORT).show();

            try{
                panoView.setImageBitmap(MediaStore.Images.Media.getBitmap(this.getContentResolver(), imagesList.get(0)));
            }
            catch (IOException e){
                e.printStackTrace();
            }
        }
    }

    public void stitchClicked(View v) throws IOException {
        //multiple stitches https://stackoverflow.com/questions/24563173/stitch-multiple-images-using-opencv-python
        //loop and calculate H
        ArrayList<Mat> HList = new ArrayList<Mat>();
        for (int i=0; i<imagesList.size()-1; i++){
            Bitmap im1 = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imagesList.get(i));
            Bitmap im2 = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imagesList.get(i+1));
            Mat img1 = new Mat();
            Mat img2 = new Mat();
            Utils.bitmapToMat(im1,img1);
            Utils.bitmapToMat(im2,img2);
            HList.add(getHomography(img1,img2));
        }
        //init img_matches = first image
        Bitmap input = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imagesList.get(0));
        img_matches = new Mat(new Size(input.getWidth()+input.getWidth(),input.getHeight()), CvType.CV_32FC2);
        Utils.bitmapToMat(input,img_matches);

        for (int i=1; i<imagesList.size(); i++){
            Bitmap im = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imagesList.get(i));
            Mat img = new Mat();
            Utils.bitmapToMat(im,img);
            Mat H = HList.get(i-1);
            //for (int j=1; j<i; j++){
            //    Core.gemm(H, HList.get(j),1,new Mat(),0,H);
            //}
            img_matches = stitchImage(img_matches,img,H);
        }

        //convert to bitmap and display
        panoBitmap = Bitmap.createBitmap(img_matches.width(),img_matches.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img_matches,panoBitmap);
        panoView.setImageBitmap(panoBitmap);
    }

    //stitch 2 inputs for 1 output
    public Mat stitchImage(Mat img1, Mat img2, Mat Homography){
        //create panorama mat
        Size s = new Size(img1.cols() + img2.cols(),img1.rows());
        Mat img_matches = new Mat(new Size(img1.cols()+img2.cols(),img1.rows()), CvType.CV_8U);
        //overlay
        Imgproc.warpPerspective(img1, img_matches, Homography, s);
        Mat m = new Mat(img_matches,new Rect(0,0,img2.cols(), img2.rows()));
        img2.copyTo(m);
        return img_matches;

    }

    //find homography matrix H of img1, img2
    public Mat getHomography(Mat img1, Mat img2){
        Mat gray_image1 = new Mat();
        Mat gray_image2 = new Mat();

        Imgproc.cvtColor(img1, gray_image1, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(img2, gray_image2, Imgproc.COLOR_RGB2GRAY);

        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();


        FeatureDetector detector = FeatureDetector.create(FeatureDetector.FAST);
        switch(Detector){
            case "FAST":
                detector = FeatureDetector.create(FeatureDetector.FAST); //orb,fast
                break;

            case "ORB":
                detector = FeatureDetector.create(FeatureDetector.ORB); //orb,fast
                break;

            case "AKAZE":
                detector = FeatureDetector.create(FeatureDetector.AKAZE); //orb,fast
                break;

            case "BRISK":
                detector = FeatureDetector.create(FeatureDetector.BRISK); //orb,fast
                break;

            case "GFTT":
                detector = FeatureDetector.create(FeatureDetector.GFTT); //orb,fast
                break;
        }

        detector.detect(gray_image1, keyPoints1);
        detector.detect(gray_image2, keyPoints2);

        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();



        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        switch(Descriptor){
            case "ORB":
                extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
                break;

            case "BRISK":
                extractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);
                break;
        }
        extractor.compute(gray_image1, keyPoints1, descriptors1);
        extractor.compute(gray_image2, keyPoints2, descriptors2);

        MatOfDMatch matches = new MatOfDMatch();

        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);

        matcher.match(descriptors1, descriptors2, matches);

        List<DMatch> listMatches = matches.toList();

        LinkedList<Point> imgPoints1List = new LinkedList<Point>();
        LinkedList<Point> imgPoints2List = new LinkedList<Point>();
        List<KeyPoint> keypoints1List = keyPoints1.toList();
        List<KeyPoint> keypoints2List = keyPoints2.toList();

        for(int i = 0; i<listMatches.size(); i++){
            imgPoints1List.addLast(keypoints1List.get(listMatches.get(i).queryIdx).pt);
            imgPoints2List.addLast(keypoints2List.get(listMatches.get(i).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(imgPoints1List);
        MatOfPoint2f scene = new MatOfPoint2f();
        scene.fromList(imgPoints2List);

        Mat H = Calib3d.findHomography(obj, scene, Calib3d.RANSAC,Integer.parseInt(RANSACvalue));

        return H;
    }

    public void saveClicked(View v){
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        String fileName = Environment.getExternalStorageDirectory().getPath() + "/DCIM/Camera/" + currentDateandTime + ".jpg";
        Imgproc.cvtColor(img_matches,img_matches,Imgproc.COLOR_RGBA2BGR);
        Imgcodecs.imwrite(fileName,img_matches);
        Toast.makeText(this, "Panorama saved", Toast.LENGTH_SHORT).show();
        scanFile(fileName); //make it appear in gallery
    }

    private void scanFile(String path) {
        MediaScannerConnection.scanFile(StitcherActivity.this,
                new String[] { path }, null,
                new MediaScannerConnection.OnScanCompletedListener() {

                    public void onScanCompleted(String path, Uri uri) {
                        Log.i("TAG", "Finished scanning " + path);
                    }
                });
    }

    public void initFeatureLists(){
        detectorList = (Spinner) findViewById(R.id.detectorList);
        List<String> listDetector = new ArrayList<>();
        listDetector.add("FAST");
        listDetector.add("ORB");
        listDetector.add("AKAZE");
        listDetector.add("BRISK");
        listDetector.add("GFTT");

        ArrayAdapter<String> adapterDetector = new ArrayAdapter(this, android.R.layout.simple_spinner_item,listDetector);
        adapterDetector.setDropDownViewResource(android.R.layout.simple_list_item_single_choice);
        detectorList.setAdapter(adapterDetector);
        detectorList.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                Detector =  detectorList.getSelectedItem().toString();
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
            }
        });

        descriptorList = (Spinner) findViewById(R.id.descriptorList);
        List<String> listDescriptor = new ArrayList<>();
        listDescriptor.add("ORB");
        listDescriptor.add("BRISK");


        ArrayAdapter<String> adapterDescriptor = new ArrayAdapter(this, android.R.layout.simple_spinner_item,listDescriptor);
        adapterDescriptor.setDropDownViewResource(android.R.layout.simple_list_item_single_choice);
        descriptorList.setAdapter(adapterDescriptor);
        descriptorList.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                Descriptor =  descriptorList.getSelectedItem().toString();
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
            }
        });

        ransacThresList = (Spinner) findViewById(R.id.ransacThresList);
        List<String> listRANSACE = new ArrayList<>();
        listRANSACE.add("5");
        listRANSACE.add("10");
        listRANSACE.add("15");
        listRANSACE.add("20");
        listRANSACE.add("25");
        listRANSACE.add("30");

        ArrayAdapter<String> RANSACAdapter = new ArrayAdapter(this, android.R.layout.simple_spinner_item,listRANSACE);
        adapterDescriptor.setDropDownViewResource(android.R.layout.simple_list_item_single_choice);
        ransacThresList.setAdapter(RANSACAdapter);
        ransacThresList.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                RANSACvalue =  ransacThresList.getSelectedItem().toString();
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
            }
        });
    }

}
