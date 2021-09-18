package dev.prosciutto.defisheye.opencv;

import java.nio.file.Path;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public final class ImageHelper {

  private ImageHelper() {
  }

  public static Collection<Mat> resolveImageMatrices(final Collection<Path> imageMatrixPaths) {
    Objects.requireNonNull(imageMatrixPaths);

    return imageMatrixPaths.stream()
        .map(ImageHelper::resolveImageMatrix)
        .filter(Optional::isPresent)
        .map(Optional::get)
        .toList();
  }

  public static Mat undistortImageMatrix(final Mat distortedImageMatrix,
      final Mat undistortionMapMatrix, final Mat rectificationMapMatrix) {
    Objects.requireNonNull(undistortionMapMatrix);
    Objects.requireNonNull(rectificationMapMatrix);

    final Mat undistortedImageMatrix = Optional.ofNullable(distortedImageMatrix)
        .map(Mat::clone)
        .orElseThrow(IllegalArgumentException::new);

    Imgproc.remap(distortedImageMatrix, undistortedImageMatrix, undistortionMapMatrix,
        rectificationMapMatrix, Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT);

    return undistortedImageMatrix;
  }

  public static void writeImageMatrix(final Mat imageMatrix, final Path path) {
    Objects.requireNonNull(imageMatrix);
    Objects.requireNonNull(path);

    Imgcodecs.imwrite(path.toString(), imageMatrix);
  }

  public static Entry<Mat, Mat> computeUndistortionAndRectificationMapMatrices(final Size size,
      final Mat cameraIntrinsicMatrix, final Mat distortionCoefficientsMatrix) {
    final Mat undistortionMapMatrix = new Mat();
    final Mat rectificationMapMatrix = new Mat();
    Calib3d.fisheye_initUndistortRectifyMap(cameraIntrinsicMatrix,
        distortionCoefficientsMatrix, Mat.eye(3, 3, CvType.CV_32F),
        cameraIntrinsicMatrix, size, CvType.CV_16SC2, undistortionMapMatrix,
        rectificationMapMatrix);
    return Map.entry(undistortionMapMatrix, rectificationMapMatrix);
  }

  public static Entry<Mat, Mat> computeCameraIntrinsicAndDistortionCoefficientsMatrices(
      final List<Mat> objectPointMatrices, final List<Mat> imagePointMatrices, final Size size) {
    final Mat cameraIntrinsicMatrix = Mat.zeros(3, 3, CvType.CV_64F);
    final Mat distortionCoefficientsMatrix = Mat.zeros(4, 1, CvType.CV_64F);
    final List<Mat> rotationVectorMatrices = IntStream.range(0, imagePointMatrices.size())
        .mapToObj(unused -> Mat.zeros(new int[]{1, 1, 3}, CvType.CV_64F))
        .collect(Collectors.toList());
    final List<Mat> translationVectorMatrices = IntStream.range(0, imagePointMatrices.size())
        .mapToObj(unused -> Mat.zeros(new int[]{1, 1, 3}, CvType.CV_64F))
        .collect(Collectors.toList());
    Calib3d.fisheye_calibrate(objectPointMatrices, imagePointMatrices, size, cameraIntrinsicMatrix,
        distortionCoefficientsMatrix, rotationVectorMatrices, translationVectorMatrices,
        Calib3d.fisheye_CALIB_RECOMPUTE_EXTRINSIC + Calib3d.fisheye_CALIB_CHECK_COND
            + Calib3d.fisheye_CALIB_FIX_SKEW,
        new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 1e-6));
    return Map.entry(cameraIntrinsicMatrix, distortionCoefficientsMatrix);
  }

  private static Optional<Mat> resolveImageMatrix(final Path path) {
    return Optional.ofNullable(path)
        .map(Path::toString)
        .map(Imgcodecs::imread);
  }
}
