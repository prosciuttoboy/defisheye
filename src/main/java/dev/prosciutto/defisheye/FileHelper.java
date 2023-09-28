package dev.prosciutto.defisheye;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.Objects;
import java.util.Optional;

final class FileHelper {

  private FileHelper() {
  }

  public static Collection<Path> listDirectory(final String directory) {
    Objects.requireNonNull(directory);

    final var uri = Optional.ofNullable(FileHelper.class.getClassLoader().getResource(directory))
        .orElseThrow(() -> new IllegalArgumentException("No such directory: %s".formatted(directory)));

    try (final var pathStream = Files.list(Paths.get(uri.toURI()))) {
      return pathStream.toList();
    } catch (final IOException | URISyntaxException exception) {
      throw new IllegalArgumentException(exception);
    }
  }

  public static Path createTemporaryFile(final String filenamePrefix,
      final String filenameSuffix) {
    Objects.requireNonNull(filenamePrefix);
    Objects.requireNonNull(filenameSuffix);

    try {
      return Files.createTempFile(filenamePrefix, filenameSuffix);
    } catch (final IOException exception) {
      throw new IllegalStateException();
    }
  }
}
