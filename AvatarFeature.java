package org.hipi.examples;

import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;
import org.hipi.imagebundle.mapreduce.HibInputFormat;


import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;


import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;


import java.io.IOException;
import java.net.URI;
import java.util.*;


public class AvatarFeature extends Configured implements Tool {


   public static class FaceCountMapper extends Mapper<HipiImageHeader, FloatImage, IntWritable, Text> {


       // Create a face detector from the cascade file in the resources
       // directory.
      private CascadeClassifier faceDetector;


       // Convert HIPI FloatImage to OpenCV Mat
       public Mat convertFloatImageToOpenCVMat(FloatImage floatImage) {


           // Get dimensions of image
           int w = floatImage.getWidth();
           int h = floatImage.getHeight();


           // Get pointer to image data
           float[] valData = floatImage.getData();


           // Initialize 3 element array to hold RGB pixel average
           double[] rgb = {0.0,0.0,0.0};


           Mat mat = new Mat(h, w, CvType.CV_8UC3);


           // Traverse image pixel data in raster-scan order and update running average
           for (int j = 0; j < h; j++) {
               for (int i = 0; i < w; i++) {
                   rgb[0] = (double) valData[(j*w+i)*3+0] * 255.0; // R
                   rgb[1] = (double) valData[(j*w+i)*3+1] * 255.0; // G
                   rgb[2] = (double) valData[(j*w+i)*3+2] * 255.0; // B
                   mat.put(j, i, rgb);
               }
           }


           return mat;
       }


       // Count faces in image
       public int countFaces(Mat image) {


           // Detect faces in the image.
           // MatOfRect is a special container class for Rect.
           MatOfRect faceDetections = new MatOfRect();
           faceDetector.detectMultiScale(image, faceDetections);


           return faceDetections.toArray().length;
       }

       
       // describe color histogram
       public String describe(Mat image) {
           Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2HSV);
            List<Mat> features = new ArrayList();
            int segments[][] = new int[4][4];

            int w = image.cols();
            int h = image.rows();
            int cX = (int)(w * 0.5);
            int cY = (int)(h * 0.5);
            
            segments[0][0] = 0; segments[0][1] = cX; segments[0][2] = 0; segments[0][3] = cY;
            segments[1][0] = cX; segments[1][1] = w; segments[1][2] = 0; segments[1][3] = cY;
            segments[2][0] = cX; segments[2][1] = w; segments[2][2] = cY; segments[2][3] = h;
            segments[3][0] = 0; segments[3][1] = cX; segments[3][2] = cY; segments[3][3] = h;
            
            int axesX = (int)((w * 0.75) / 2);
            int axesY = (int)((h * 0.75) / 2);

            Mat ellipMask = new Mat(h, w, CvType.CV_8UC1);
            ellipMask.setTo(new Scalar(0.0), ellipMask);
            Core.ellipse(ellipMask, new Point(cX,cY), new Size(axesX, axesY), 0, 0, 360, new Scalar(255), -1);
            
            Mat hist = new Mat();
            String strFeatures = "";
            strFeatures = strFeatures + "ic:" + String.valueOf(image.cols()) + ",ir:" + String.valueOf(image.rows());
            for (int i = 0; i < 4; i++) {
                Mat cornerMask = new Mat(h, w, CvType.CV_8UC1);
                cornerMask.setTo(new Scalar(0.0), cornerMask);
                Core.rectangle(cornerMask, new Point(segments[i][0], segments[i][1]), new Point(segments[i][2], segments[i][3]), new Scalar(255), -1);
                Core.subtract(cornerMask, ellipMask, cornerMask);
                    
                //hist = histogram(image, cornerMask);
                strFeatures = strFeatures + "," + histogram(image, cornerMask);
                //strFeatures = strFeatures + "hc," + String.valueOf(hist.cols()) + "hr," + String.valueOf(hist.rows()); 
                features.add(hist);
            }
            strFeatures = strFeatures + "," + histogram(image, ellipMask);
            //strFeatures = strFeatures + "ec," + String.valueOf(hist.cols()) + "er," + String.valueOf(hist.rows());
            //features.add(hist);
            return strFeatures;
       }
       
       public String histogram(Mat image, Mat mask) {
            List<Mat> histImages = new ArrayList<Mat>();
            histImages.add(image);
            MatOfInt channels = new MatOfInt(0, 1, 2);
            
            
            Mat h_hist = new Mat();
        
            MatOfInt h_size = new MatOfInt(8, 12, 3);
            MatOfFloat ranges = new MatOfFloat(0.0f, 180.0f, 0.0f, 255.0f, 0f, 255.0f);
            Imgproc.calcHist(histImages, new MatOfInt(0), mask, h_hist, new MatOfInt(8), new MatOfFloat(0.0f, 180.0f));
    
   
            Core.normalize(h_hist, h_hist);
            
            String strHist = "";
            int sizeh = (int) h_hist.total() * h_hist.channels();
            float[] buffh = new float[sizeh];
            h_hist.get(0, 0, buffh);
            for (int i = 0; i < sizeh; i++) {
                strHist = strHist + " " + String.valueOf(buffh[i]);
            }
            
            return strHist;
        }   

       
       public void setup(Context context)
               throws IOException, InterruptedException {


           // Load OpenCV native library
           try {
               System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
           } catch (UnsatisfiedLinkError e) {
               System.err.println("Native code library failed to load.\n" + e + Core.NATIVE_LIBRARY_NAME);
               System.exit(1);
           }


           // Load cached cascade file for front face detection and create CascadeClassifier
           if (context.getCacheFiles() != null && context.getCacheFiles().length > 0) {
               URI mappingFileUri = context.getCacheFiles()[0];


               if (mappingFileUri != null) {
                   faceDetector = new CascadeClassifier("lbpcascade_frontalface.xml");


               } else {
                   System.out.println(">>>>>> NO MAPPING FILE");
               }
           } else {
               System.out.println(">>>>>> NO CACHE FILES AT ALL");
           }


           super.setup(context);
       } // setup()


       public void map(HipiImageHeader key, FloatImage value, Context context)
               throws IOException, InterruptedException {


           // Verify that image was properly decoded, is of sufficient size, and has three color channels (RGB)
           if (value != null && value.getWidth() > 1 && value.getHeight() > 1 && value.getNumBands() == 3) {

               String filename = key.getMetaData("filename");
               Mat cvImage = this.convertFloatImageToOpenCVMat(value);


               //int faces = this.countFaces(cvImage);
               //
               //
               //
               String features = describe(cvImage);
               //filename = filename + "," + String.valueOf(features.size());
               filename = filename + "," + features;
            /*   for (int row = 0; row < cvImage.rows(); row++) {
                   for (int col = 0; col < cvImage.cols(); col++) {
                       filename = filename + "," + String.valueOf(cvImage.get(row, col));
                   }
               }
               
               Mat temp = new Mat();
               for (int i = 0; i < features.size(); i++) {
                   temp = features.get(i);
                   filename = filename + "," + String.valueOf(temp.rows()) + "," + String.valueOf(temp.cols());
                   for (int row = 0; row < temp.rows(); row++) {
                       for (int col = 0; col < temp.cols(); col++) {
                           filename = filename + "," + String.valueOf(temp.get(row, col));
                       }
                   }
               }
             */
               
               
               System.out.println(">>>>>> image features: ");
               context.write(new IntWritable(1), new Text(filename));
               //System.out.println(">>>>>> Detected Faces: " + Integer.toString(faces));


               // Emit record to reducer
               //context.write(new IntWritable(1), new IntWritable(faces));


           } // If (value != null...


       } // map()
   }


   //public static class FaceCountReducer extends Reducer<IntWritable, IntWritable, IntWritable, Text> {
   public static class FaceCountReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
       //public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
       public void reduce(IntWritable key, Iterable<Text> values, Context context)
               throws IOException, InterruptedException {


            // Initialize a counter and iterate over IntWritable/FloatImage records from mapper
           int total = 0;
           int images = 0;
           
           String output = "";
           //for (IntWritable val : values) {
           for (Text val : values) {
               //total += val.get();
               output = output + val.toString() + "\n";
               
               images++;
           }


           //String result = String.format("Total face detected: %d", total);
           //String result = String.format("%s", output);
           // Emit output of job which will be written to HDFS
           //context.write(new IntWritable(images), new Text(result));
           context.write(new IntWritable(images), new Text(output));
       } // reduce()
   }


   public int run(String[] args) throws Exception {
       // Check input arguments
       if (args.length != 2) {
           System.out.println("Usage: firstprog <input HIB> <output directory>");
           System.exit(0);
       }


       // Initialize and configure MapReduce job
       Job job = Job.getInstance();
       // Set input format class which parses the input HIB and spawns map tasks
       job.setInputFormatClass(HibInputFormat.class);
       // Set the driver, mapper, and reducer classes which express the computation
       job.setJarByClass(FaceCount.class);
       job.setMapperClass(FaceCountMapper.class);
       job.setReducerClass(FaceCountReducer.class);
       // Set the types for the key/value pairs passed to/from map and reduce layers
       job.setMapOutputKeyClass(IntWritable.class);
       
       //job.setMapOutputValueClass(IntWritable.class);
       job.setMapOutputValueClass(Text.class);
       
       job.setOutputKeyClass(IntWritable.class);
       job.setOutputValueClass(Text.class);


       // Set the input and output paths on the HDFS
       FileInputFormat.setInputPaths(job, new Path(args[0]));
       FileOutputFormat.setOutputPath(job, new Path(args[1]));


       // add cascade file
       job.addCacheFile(new URI("hdfs:///lbpcascade_frontalface.xml"));


       // Execute the MapReduce job and block until it complets
       boolean success = job.waitForCompletion(true);


       // Return success or failure
       return success ? 0 : 1;
   }


   public static void main(String[] args) throws Exception {


       ToolRunner.run(new FaceCount(), args);
       System.exit(0);
   }


}