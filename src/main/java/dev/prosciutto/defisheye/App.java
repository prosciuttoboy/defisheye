package dev.prosciutto.defisheye;

import dev.prosciutto.defisheye.io.FileHelper;
import dev.prosciutto.defisheye.opencv.ImageHelper;
import java.util.ArrayList;
import java.util.stream.IntStream;
import nu.pattern.OpenCV;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

public final class App {

  private static final Size CHECKERBOARD = new Size(6, 9);

  static {
    OpenCV.loadLocally();
  }

  public static void main(final String[] args) {
    final MatOfPoint3f objectPoint = new MatOfPoint3f(IntStream.range(0, (int) CHECKERBOARD.area())
        .mapToObj(squareIndex -> new Point3(squareIndex % (int) CHECKERBOARD.width,
            squareIndex / (int) CHECKERBOARD.width, 0))
        .toArray(Point3[]::new));
    final var objectPoints = new ArrayList<Mat>();
    final var imagePoints = new ArrayList<Mat>();
    final var referenceImage = new Mat();
    ImageHelper.resolveImageMatrices(FileHelper.listDirectory("calibration"))
        .stream()
        .map(calibrationImageMatrix -> {
          if (referenceImage.empty()) {
            calibrationImageMatrix.copyTo(referenceImage);
          }
          final Mat newImage = calibrationImageMatrix.clone();
          Imgproc.cvtColor(calibrationImageMatrix, newImage, Imgproc.COLOR_BGR2GRAY);
          return newImage;
        })
        .forEach(greyImage -> {
          final MatOfPoint2f corners = new MatOfPoint2f();
          Calib3d.findChessboardCorners(greyImage, CHECKERBOARD, corners,
              Calib3d.CALIB_CB_ADAPTIVE_THRESH + Calib3d.CALIB_CB_FAST_CHECK
                  + Calib3d.CALIB_CB_NORMALIZE_IMAGE);
          if (!corners.empty()) {
            objectPoints.add(objectPoint);
            Imgproc.cornerSubPix(greyImage, corners, new Size(3, 3), new Size(-1, -1),
                new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.1));
            imagePoints.add(corners);
          }
        });
    final var cameraIntrinsicAndDistortionCoefficientsMatrices = ImageHelper.computeCameraIntrinsicAndDistortionCoefficientsMatrices(
        objectPoints, imagePoints, referenceImage.size());
    final var undistortionAndRectificationMapMatrices = ImageHelper.computeUndistortionAndRectificationMapMatrices(
        referenceImage.size(), cameraIntrinsicAndDistortionCoefficientsMatrices.getKey(),
        cameraIntrinsicAndDistortionCoefficientsMatrices.getValue());
    ImageHelper.resolveImageMatrices(FileHelper.listDirectory("candidate"))
        .stream()
        .map(distortedImageMatrix -> {
          final var undistortedImageMatrix = ImageHelper.undistortImageMatrix(distortedImageMatrix,
              undistortionAndRectificationMapMatrices.getKey(),
              undistortionAndRectificationMapMatrices.getValue());
          final var undistortedImageMatrixPath = FileHelper.createTemporaryFile("remapped", ".jpg");
          ImageHelper.writeImageMatrix(undistortedImageMatrix, undistortedImageMatrixPath);
          return undistortedImageMatrixPath;
        })
        .forEach(System.out::println);
  }
}
