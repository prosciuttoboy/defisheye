package dev.prosciutto.defisheye;

import dev.prosciutto.defisheye.io.FileHelper;
import dev.prosciutto.defisheye.opencv.ImageHelper;
import java.util.ArrayList;
import java.util.Optional;
import java.util.stream.IntStream;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;
import org.opencv.core.Size;

public final class App {

  private static final Size CHESSRBOARD = new Size(6, 9);

  static {
    OpenCV.loadLocally();
  }

  public static void main(final String[] args) {
    final MatOfPoint3f objectPoint = new MatOfPoint3f(IntStream.range(0, (int) CHESSRBOARD.area())
        .mapToObj(squareIndex -> new Point3(squareIndex % (int) CHESSRBOARD.width,
            squareIndex / (int) CHESSRBOARD.width, 0))
        .toArray(Point3[]::new));
    final var objectPoints = new ArrayList<Mat>();
    final var imagePoints = new ArrayList<Mat>();
    final var referenceImageMatrix = new Mat();
    ImageHelper.resolveImageMatrices(FileHelper.listDirectory("calibration"))
        .stream()
        .peek(calibrationImageMatrix -> {
          if (referenceImageMatrix.empty()) {
            calibrationImageMatrix.copyTo(referenceImageMatrix);
          }
        })
        .map(calibrationImageMatrix -> ImageHelper.computeChessboardCorners(calibrationImageMatrix,
            CHESSRBOARD))
        .filter(Optional::isPresent)
        .map(Optional::get)
        .forEach(chessboardCornersMatrix -> {
          objectPoints.add(objectPoint);
          imagePoints.add(chessboardCornersMatrix);
        });
    final var cameraIntrinsicAndDistortionCoefficientsMatrices = ImageHelper.computeCameraIntrinsicAndDistortionCoefficientsMatrices(
        objectPoints, imagePoints, referenceImageMatrix.size());
    final var estimatedCameraIntrinsicMatrix = ImageHelper.estimateCameraIntrinsicMatrix(
        referenceImageMatrix.size(), cameraIntrinsicAndDistortionCoefficientsMatrices.getKey(),
        cameraIntrinsicAndDistortionCoefficientsMatrices.getValue());
    final var undistortionAndRectificationMapMatrices = ImageHelper.computeUndistortionAndRectificationMapMatrices(
        referenceImageMatrix.size(), cameraIntrinsicAndDistortionCoefficientsMatrices.getKey(),
        cameraIntrinsicAndDistortionCoefficientsMatrices.getValue(),
        estimatedCameraIntrinsicMatrix);
    ImageHelper.resolveImageMatrices(FileHelper.listDirectory("candidate"))
        .stream()
        .map(distortedImageMatrix -> {
          final var undistortedImageMatrix = ImageHelper.undistortImageMatrix(distortedImageMatrix,
              undistortionAndRectificationMapMatrices.getKey(),
              undistortionAndRectificationMapMatrices.getValue());
          final var undistortedImageMatrixPath = FileHelper.createTemporaryFile("undistorted",
              ".jpg");
          ImageHelper.writeImageMatrix(undistortedImageMatrix, undistortedImageMatrixPath);
          return undistortedImageMatrixPath;
        })
        .forEach(System.out::println);
  }
}
