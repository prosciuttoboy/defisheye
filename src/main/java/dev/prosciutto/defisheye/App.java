package dev.prosciutto.defisheye;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import nu.pattern.OpenCV;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class App {

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
    readImages("calibration").map(originalImage -> {
          if (referenceImage.empty()) {
            originalImage.copyTo(referenceImage);
          }
          final Mat newImage = originalImage.clone();
          Imgproc.cvtColor(originalImage, newImage, Imgproc.COLOR_BGR2GRAY);
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
    final Mat K = Mat.zeros(3, 3, CvType.CV_64F);
    final Mat D = Mat.zeros(4, 1, CvType.CV_64F);
    final List<Mat> rvecs = new ArrayList<>();
    final List<Mat> tvecs = new ArrayList<>();
    Calib3d.fisheye_calibrate(objectPoints, imagePoints, referenceImage.size(), K, D, rvecs, tvecs,
        Calib3d.fisheye_CALIB_RECOMPUTE_EXTRINSIC + Calib3d.fisheye_CALIB_CHECK_COND
            + Calib3d.fisheye_CALIB_FIX_SKEW,
        new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 1e-6));
    readImages("candidate")
        .forEach(candidateImage -> {
          final Mat map1 = new Mat();
          final Mat map2 = new Mat();
          Calib3d.fisheye_initUndistortRectifyMap(K, D, Mat.eye(3, 3, CvType.CV_64F), K,
              referenceImage.size(), CvType.CV_16SC2, map1, map2);
          final Mat undistortedImage = candidateImage.clone();
          Imgproc.remap(candidateImage, undistortedImage, map1, map2, Imgproc.INTER_LINEAR,
              Core.BORDER_CONSTANT);
          try {
            final var undistortedImageFilePath = Files.createTempFile("undistorted", ".jpg");
            Imgcodecs.imwrite(undistortedImageFilePath.toString(), undistortedImage);
            System.out.println(undistortedImageFilePath);
          } catch (final IOException exception) {
            throw new RuntimeException(exception);
          }
        });

  }

  private static Stream<Mat> readImages(final String directory) {
    try {
      final var uri = Optional.ofNullable(App.class.getClassLoader().getResource(directory))
          .map(url -> {
            try {
              return url.toURI();
            } catch (final URISyntaxException exception) {
              throw new RuntimeException(exception);
            }
          })
          .orElseThrow();
      return Files.list(Paths.get(uri))
          .map(Path::toString)
          .map(Imgcodecs::imread)
          .filter(mat -> !mat.empty());
    } catch (final IOException exception) {
      throw new RuntimeException(exception);
    }
  }
}
