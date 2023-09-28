package dev.prosciutto.defisheye;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacv.FFmpegFrameFilter;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_calib3d;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.Point2fVector;
import org.bytedeco.opencv.opencv_core.Point2fVectorVector;
import org.bytedeco.opencv.opencv_core.Point3f;
import org.bytedeco.opencv.opencv_core.Point3fVector;
import org.bytedeco.opencv.opencv_core.Point3fVectorVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.TermCriteria;

final class JavaCvCameraCalibration implements CameraCalibration {

  private final CameraCalibrationConfiguration cameraCalibrationConfiguration;
  private final Mat undistortionMapMatrix = new Mat();
  private final Mat rectificationMapMatrix = new Mat();

  public JavaCvCameraCalibration(final CameraCalibrationConfiguration cameraCalibrationConfiguration) {
    this.cameraCalibrationConfiguration = cameraCalibrationConfiguration;
    try (final var ignored = new PointerScope()) {
      final var imageMats = resolveImageMats();
      validateImageMats(imageMats);
      final var imagePoints = resolveImagePoints(imageMats);
      final var imageSize = imageMats.get(0).size();
      imageMats.forEach(Pointer::close);
      validateImagePoints(imagePoints);
      setUndistortionMapAndRectificationMapMats(imageSize, imagePoints);
    }
  }

  private List<Mat> resolveImageMats() {
    return cameraCalibrationConfiguration.images()
        .stream()
        .map(this::resolveImageMat)
        .toList();
  }

  private Mat resolveImageMat(final Path source) {
    return Optional.of(opencv_imgcodecs.imread(source.toString()))
        .filter(Predicate.not(Mat::empty))
        .orElseThrow();
  }

  private void validateImageMats(final List<Mat> imageMats) {
    if (imageMats.isEmpty()) {
      throw new IllegalArgumentException();
    }
    try (final var imageSize = imageMats.get(0).size()) {
      if (imageMats.stream()
          .skip(1L)
          .anyMatch(imageMat -> {
            try (final var imageMatSize = imageMat.size()) {
              return imageMatSize.width() != imageSize.width()
                  && imageMatSize.height() != imageSize.height();
            }
          })) {
        throw new IllegalArgumentException();
      }
    }
  }

  private Point2fVectorVector resolveImagePoints(final List<Mat> imageMats) {
    try (final var chessboardSize = new Size(cameraCalibrationConfiguration.chessboardWidth(), cameraCalibrationConfiguration.chessboardHeight());
        final var winSize = new Size(3, 3);
        final var zeroZone = new Size(-1, -1);
        final var termCriteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.1)) {
      final var chessboardCalibrationFlags =
          opencv_calib3d.CALIB_CB_ADAPTIVE_THRESH + opencv_calib3d.CALIB_CB_FAST_CHECK + opencv_calib3d.CALIB_CB_NORMALIZE_IMAGE;
      final var chessboardCornersPoints = new ArrayList<Point2fVector>();

      for (final var imageMat : imageMats) {
        try (final var grayscaleImageMat = new Mat();
            final var chessboardCornersMat = new Mat()) {
          opencv_imgproc.cvtColor(imageMat, grayscaleImageMat, opencv_imgproc.COLOR_BGR2GRAY);
          opencv_calib3d.findChessboardCorners(grayscaleImageMat, chessboardSize, chessboardCornersMat, chessboardCalibrationFlags);
          if (!chessboardCornersMat.empty()) {
            opencv_imgproc.cornerSubPix(grayscaleImageMat, chessboardCornersMat, winSize, zeroZone, termCriteria);
            try (final var chessboardCornersSize = chessboardCornersMat.size()) {
              if (chessboardCornersSize.area() == chessboardSize.area()) {
                try (final var indexer = chessboardCornersMat.<FloatIndexer>createIndexer()) {
                  chessboardCornersPoints.add(new Point2fVector(LongStream.range(0, chessboardCornersMat.total())
                      .mapToObj(i -> new Point2f(indexer.get(0, i, 0), indexer.get(0, i, 1)))
                      .toArray(Point2f[]::new)));
                }
              }
            }
          }
        }
      }

      return new Point2fVectorVector(chessboardCornersPoints.toArray(Point2fVector[]::new));
    }
  }

  private void validateImagePoints(final Point2fVectorVector imagePoints) {
    if (imagePoints.empty()) {
      throw new IllegalArgumentException();
    }
  }

  private void setUndistortionMapAndRectificationMapMats(final Size imageSize, final Point2fVectorVector imagePoints) {
    final var cameraIntrinsicMat = new Mat();
    final var distortionCoefficientsMat = new Mat();
    try (final var objectPoints = new Point3fVectorVector(LongStream.range(0, imagePoints.size())
        .mapToObj(unused -> new Point3fVector(
            IntStream.range(0, cameraCalibrationConfiguration.chessboardWidth() * cameraCalibrationConfiguration.chessboardHeight())
                .mapToObj(
                    i -> new Point3f(i % cameraCalibrationConfiguration.chessboardWidth(), i / cameraCalibrationConfiguration.chessboardWidth(), 0))
                .toArray(Point3f[]::new)))
        .toArray(Point3fVector[]::new));
        imagePoints;
        final var rvec = new MatVector();
        final var tvecs = new MatVector();
        final var termCriteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 1e-6)) {
      opencv_calib3d.fisheyeCalibrate(objectPoints, imagePoints, imageSize, cameraIntrinsicMat, distortionCoefficientsMat, rvec, tvecs,
          opencv_calib3d.FISHEYE_CALIB_RECOMPUTE_EXTRINSIC + opencv_calib3d.FISHEYE_CALIB_CHECK_COND + opencv_calib3d.FISHEYE_CALIB_FIX_SKEW,
          termCriteria);
    }
    try (final var eyeMatExpr = Mat.eye(3, 3, opencv_core.CV_32F);
        final var rectificationTransformationMat = eyeMatExpr.asMat();
        final var estimatedCameraIntrinsicMat = new Mat()) {
      opencv_calib3d.fisheyeEstimateNewCameraMatrixForUndistortRectify(cameraIntrinsicMat, distortionCoefficientsMat, imageSize,
          rectificationTransformationMat, estimatedCameraIntrinsicMat);
      opencv_calib3d.fisheyeInitUndistortRectifyMap(cameraIntrinsicMat, distortionCoefficientsMat, rectificationTransformationMat,
          estimatedCameraIntrinsicMat, imageSize, opencv_core.CV_16SC2, undistortionMapMatrix, rectificationMapMatrix);
    }
  }

  @Override
  public void undistortImage(final Path source, final Path destination) {
    try (final var imageMat = resolveImageMat(source)) {
      undistortImage(imageMat);
      opencv_imgcodecs.imwrite(destination.toString(), imageMat);
    }
  }

  @Override
  public void undistortVideo(final Path source, final Path destination) {
    try (final var frameGrabber = FFmpegFrameGrabber.createDefault(source.toString());
        final var frameConverter = new OpenCVFrameConverter.ToMat()) {
      frameGrabber.start();
      final var imageWidth = frameGrabber.getImageWidth();
      final var imageHeight = frameGrabber.getImageHeight();
      try (final var frameRecorder = FFmpegFrameRecorder.createDefault(destination.toString(), imageWidth, imageHeight);
          final var frameFilter = new FFmpegFrameFilter("[in] tmix=3 [out]; [out] framestep=step=6", imageWidth, imageHeight)) {
        frameRecorder.setVideoQuality(0);
        frameRecorder.setFrameRate(5);
        frameRecorder.start();
        frameFilter.start();
        Frame currentFrame;
        while ((currentFrame = frameGrabber.grabImage()) != null) {
          final var imageMat = frameConverter.convert(currentFrame);
          undistortImage(imageMat);
          try (final var undistortedFrame = frameConverter.convert(imageMat)) {
            frameFilter.push(undistortedFrame);
            Frame filteredFrame;
            while ((filteredFrame = frameFilter.pull()) != null) {
              frameRecorder.record(filteredFrame);
              filteredFrame.close();
            }
          }
          currentFrame.close();
        }
      }
    } catch (final IOException exception) {
      throw new IllegalStateException(exception);
    }
  }

  private void undistortImage(final Mat imageMat) {
    opencv_imgproc.remap(imageMat, imageMat, undistortionMapMatrix, rectificationMapMatrix, opencv_imgproc.INTER_LINEAR);
  }
}
