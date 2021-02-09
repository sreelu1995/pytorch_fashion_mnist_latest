package com.example.pytorchfashionmnist;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import static java.lang.StrictMath.exp;
import static java.lang.StrictMath.getExponent;

public class MainActivity extends AppCompatActivity {

    TextView mTvPrediction;
    TextView mTvProbability;
    TextView mTvTimeCost;
    private CameraView cameraView;
    private Button btnToggleCamera, btn_detect, clear;
    Bitmap bitmap, temp_bitmap;

    private final String[] class_names = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

    private static final int INPUT_SIZE = 28;


    @Override
    protected void onCreate( Bundle savedInstanceState ) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraView = findViewById(R.id.cameraView);
        btnToggleCamera = findViewById(R.id.btnToggleCamera);
        btn_detect = findViewById(R.id.btn_detect);
        clear = findViewById(R.id.btn_clear);
        mTvPrediction = findViewById(R.id.prediction);
        mTvProbability = findViewById(R.id.probability);
        mTvTimeCost = findViewById(R.id.timecost);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent( CameraKitEvent cameraKitEvent) {

            }

            @Override
            public void onError( CameraKitError cameraKitError) {

            }

            @Override
            public void onImage( CameraKitImage cameraKitImage) {

                Module module = null;

                try {
                    temp_bitmap = cameraKitImage.getBitmap();
                    bitmap = toGrayscale(temp_bitmap);
                    bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
                    bitmap = resizeBitmap(bitmap, 28);

                    module = Module.load(fetchModelFile(MainActivity.this, "scripted_model_final_RGB.pt"));
                }  catch (IOException e) {
                      finish();
                }

            final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
                        bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB
                );

                long startTime = SystemClock.uptimeMillis();
                //Calling the forward of the model to run our input
                final Tensor output = module.forward(IValue.from(input)).toTensor();

                long endTime = SystemClock.uptimeMillis();
                long timeCost = endTime - startTime;

                final float[] score_arr = output.getDataAsFloatArray();

                double max_score = 0;
                int ms_ix = -1;
                for (int i = 0; i < score_arr.length; i++) {
                    if (exp(score_arr[i]) > max_score) {
                        max_score = exp(score_arr[i]);
                        ms_ix = i;
                    }
                }

                //Fetching the name from the list based on the index
                String detected_class = class_names[ms_ix];

                //Writing the detected class in to the text view of the layout
                mTvPrediction.setText("Prediction: "+detected_class+" ");
                mTvProbability.setText("Probability: "+max_score);
                mTvTimeCost.setText("TimeCost: "+timeCost+"");
            }

            @Override
            public void onVideo( CameraKitVideo cameraKitVideo) {

            }
        });

        btnToggleCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                cameraView.toggleFacing();
            }
        });

        btn_detect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                cameraView.captureImage();
            }
        });

        clear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mTvPrediction.setText("");
                mTvProbability.setText("");
                mTvTimeCost.setText("");

            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    protected void onPause() {
        cameraView.stop();
        super.onPause();
    }

    public static String fetchModelFile( Context context, String modelName) throws IOException {
        File file = new File(context.getFilesDir(), modelName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(modelName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[9408];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    public Bitmap resizeBitmap(Bitmap getBitmap, int maxSize) {
        int width = getBitmap.getWidth();
        int height = getBitmap.getHeight();
        double x;

        if (width >= height && width > maxSize) {
            x = width / height;
            width = maxSize;
            height = (int) (maxSize / x);
        } else if (height >= width && height > maxSize) {
            x = height / width;
            height = maxSize;
            width = (int) (maxSize / x);
        }
        return Bitmap.createScaledBitmap(getBitmap, width, height, false);
    }
}
