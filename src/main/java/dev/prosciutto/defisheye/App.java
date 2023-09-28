package dev.prosciutto.defisheye;

public final class App {

  public static void main(final String[] args) {
    final var cameraCalibration = new JavaCvCameraCalibration(
        new CameraCalibrationConfiguration(FileHelper.listDirectory("calibration"), 6, 9));
    FileHelper.listDirectory("candidate/image")
        .forEach(source -> {
          final var destination = FileHelper.createTemporaryFile(source.getFileName()
              .toString(), ".jpg");
          cameraCalibration.undistortImage(source, destination);
          System.out.println(destination);
        });
    FileHelper.listDirectory("candidate/video")
        .forEach(source -> {
          final var destination = FileHelper.createTemporaryFile(source.getFileName()
              .toString(), ".mp4");
          cameraCalibration.undistortVideo(source, destination);
          System.out.println(destination);
        });
  }
}
