import java.awt.Graphics2D;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

/**
 * Athena AI — Perceptual image hashing for detecting
 * non-consensual AI-generated imagery.
 *
 * Implements DCT-based perceptual hashing (pHash) that is robust
 * to resizing, compression, cropping, and minor edits. Used to
 * match found images against enrolled reference photos.
 */
public class ImageHasher {

    private static final int HASH_SIZE = 8;      // 8x8 = 64-bit hash
    private static final int RESIZE_DIM = 32;     // Resize to 32x32 for DCT

    /**
     * Result of comparing two image hashes.
     */
    public record MatchResult(
            long hash1,
            long hash2,
            int hammingDistance,
            double similarity,
            boolean isMatch
    ) {
        @Override
        public String toString() {
            return String.format(
                    "MatchResult{similarity=%.1f%%, distance=%d, match=%s}",
                    similarity * 100, hammingDistance, isMatch
            );
        }
    }

    /**
     * Compute perceptual hash of an image file.
     *
     * @param imagePath path to the image file
     * @return 64-bit perceptual hash
     * @throws IOException if the image cannot be read
     */
    public static long computeHash(Path imagePath) throws IOException {
        BufferedImage image = ImageIO.read(imagePath.toFile());
        if (image == null) {
            throw new IOException("Could not decode image: " + imagePath);
        }
        return computeHash(image);
    }

    /**
     * Compute perceptual hash of a BufferedImage.
     *
     * @param image the input image
     * @return 64-bit perceptual hash
     */
    public static long computeHash(BufferedImage image) {
        // Step 1: Convert to grayscale
        BufferedImage gray = toGrayscale(image);

        // Step 2: Resize to 32x32
        BufferedImage resized = resize(gray, RESIZE_DIM, RESIZE_DIM);

        // Step 3: Extract pixel values as doubles
        double[][] pixels = extractPixels(resized);

        // Step 4: Compute 2D DCT
        double[][] dct = dct2D(pixels);

        // Step 5: Take top-left 8x8 (low frequency components)
        double[] lowFreq = new double[HASH_SIZE * HASH_SIZE];
        int idx = 0;
        for (int i = 0; i < HASH_SIZE; i++) {
            for (int j = 0; j < HASH_SIZE; j++) {
                lowFreq[idx++] = dct[i][j];
            }
        }

        // Step 6: Compute median
        double median = computeMedian(lowFreq.clone());

        // Step 7: Generate hash: 1 if above median, 0 if below
        long hash = 0;
        for (double val : lowFreq) {
            hash = (hash << 1) | (val > median ? 1 : 0);
        }

        return hash;
    }

    /**
     * Compute Hamming distance between two hashes.
     * Lower distance = more similar images.
     *
     * @return number of differing bits (0-64)
     */
    public static int hammingDistance(long hash1, long hash2) {
        long xor = hash1 ^ hash2;
        return Long.bitCount(xor);
    }

    /**
     * Compute similarity score between two hashes.
     *
     * @return similarity from 0.0 (completely different) to 1.0 (identical)
     */
    public static double similarity(long hash1, long hash2) {
        int distance = hammingDistance(hash1, hash2);
        return 1.0 - (distance / 64.0);
    }

    /**
     * Compare two images and return a detailed match result.
     *
     * @param threshold similarity threshold for a match (e.g. 0.75)
     */
    public static MatchResult compare(Path image1, Path image2, double threshold)
            throws IOException {
        long hash1 = computeHash(image1);
        long hash2 = computeHash(image2);
        int distance = hammingDistance(hash1, hash2);
        double sim = 1.0 - (distance / 64.0);
        return new MatchResult(hash1, hash2, distance, sim, sim >= threshold);
    }

    /**
     * Scan a directory of images against a set of reference hashes.
     *
     * @param scanDir    directory to scan
     * @param refHashes  list of reference perceptual hashes
     * @param threshold  similarity threshold for flagging
     * @return list of match results that exceed the threshold
     */
    public static List<MatchResult> scanDirectory(
            Path scanDir, List<Long> refHashes, double threshold) throws IOException {

        List<MatchResult> matches = new ArrayList<>();
        String[] extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"};

        Files.walk(scanDir)
                .filter(Files::isRegularFile)
                .filter(p -> {
                    String name = p.toString().toLowerCase();
                    for (String ext : extensions) {
                        if (name.endsWith(ext)) return true;
                    }
                    return false;
                })
                .forEach(imagePath -> {
                    try {
                        long imageHash = computeHash(imagePath);
                        for (long refHash : refHashes) {
                            double sim = similarity(imageHash, refHash);
                            if (sim >= threshold) {
                                matches.add(new MatchResult(
                                        refHash, imageHash,
                                        hammingDistance(refHash, imageHash),
                                        sim, true
                                ));
                                System.out.printf("  MATCH: %s (%.1f%% similar)%n",
                                        imagePath.getFileName(), sim * 100);
                            }
                        }
                    } catch (IOException e) {
                        System.err.println("  Skipping: " + imagePath + " (" + e.getMessage() + ")");
                    }
                });

        return matches;
    }

    /**
     * Compute SHA-256 file hash for exact duplicate detection.
     */
    public static String fileHash(Path path) throws IOException, NoSuchAlgorithmException {
        byte[] data = Files.readAllBytes(path);
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] hash = digest.digest(data);
        StringBuilder hex = new StringBuilder();
        for (byte b : hash) {
            hex.append(String.format("%02x", b));
        }
        return hex.toString();
    }

    /**
     * Convert hash to hexadecimal string.
     */
    public static String hashToHex(long hash) {
        return String.format("%016x", hash);
    }

    // ---- Internal image processing methods ----

    private static BufferedImage toGrayscale(BufferedImage image) {
        ColorConvertOp op = new ColorConvertOp(
                ColorSpace.getInstance(ColorSpace.CS_GRAY), null);
        BufferedImage gray = new BufferedImage(
                image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        op.filter(image, gray);
        return gray;
    }

    private static BufferedImage resize(BufferedImage image, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, image.getType());
        Graphics2D g = resized.createGraphics();
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    private static double[][] extractPixels(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        double[][] pixels = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                pixels[y][x] = image.getRGB(x, y) & 0xFF;
            }
        }
        return pixels;
    }

    /**
     * 2D Discrete Cosine Transform.
     */
    private static double[][] dct2D(double[][] matrix) {
        int n = matrix.length;
        double[][] result = new double[n][n];

        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                double sum = 0.0;
                for (int x = 0; x < n; x++) {
                    for (int y = 0; y < n; y++) {
                        sum += matrix[x][y]
                                * Math.cos(Math.PI * u * (2 * x + 1) / (2.0 * n))
                                * Math.cos(Math.PI * v * (2 * y + 1) / (2.0 * n));
                    }
                }
                double cu = (u == 0) ? 1.0 / Math.sqrt(n) : Math.sqrt(2.0 / n);
                double cv = (v == 0) ? 1.0 / Math.sqrt(n) : Math.sqrt(2.0 / n);
                result[u][v] = cu * cv * sum;
            }
        }

        return result;
    }

    private static double computeMedian(double[] values) {
        java.util.Arrays.sort(values);
        int mid = values.length / 2;
        if (values.length % 2 == 0) {
            return (values[mid - 1] + values[mid]) / 2.0;
        }
        return values[mid];
    }

    // ---- Main ----

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Athena AI — Image Hash Comparator");
            System.out.println();
            System.out.println("Usage:");
            System.out.println("  java ImageHasher compare <image1> <image2>");
            System.out.println("  java ImageHasher hash <image>");
            System.out.println("  java ImageHasher scan <directory> <reference_image>");
            System.exit(0);
        }

        String command = args[0];

        try {
            switch (command) {
                case "hash" -> {
                    Path path = Path.of(args[1]);
                    long hash = computeHash(path);
                    System.out.printf("File:  %s%n", path.getFileName());
                    System.out.printf("pHash: %s%n", hashToHex(hash));
                    System.out.printf("SHA256: %s%n", fileHash(path));
                }
                case "compare" -> {
                    if (args.length < 3) {
                        System.err.println("Usage: java ImageHasher compare <image1> <image2>");
                        System.exit(1);
                    }
                    MatchResult result = compare(Path.of(args[1]), Path.of(args[2]), 0.75);
                    System.out.printf("Image 1: %s (hash: %s)%n", args[1], hashToHex(result.hash1()));
                    System.out.printf("Image 2: %s (hash: %s)%n", args[2], hashToHex(result.hash2()));
                    System.out.printf("Hamming distance: %d / 64%n", result.hammingDistance());
                    System.out.printf("Similarity: %.1f%%%n", result.similarity() * 100);
                    System.out.printf("Match (>75%%): %s%n", result.isMatch() ? "YES" : "NO");
                }
                case "scan" -> {
                    if (args.length < 3) {
                        System.err.println("Usage: java ImageHasher scan <directory> <reference>");
                        System.exit(1);
                    }
                    long refHash = computeHash(Path.of(args[2]));
                    System.out.printf("Reference hash: %s%n", hashToHex(refHash));
                    System.out.printf("Scanning: %s%n%n", args[1]);
                    List<MatchResult> results = scanDirectory(
                            Path.of(args[1]), List.of(refHash), 0.75);
                    System.out.printf("%n%d match(es) found.%n", results.size());
                }
                default -> System.err.println("Unknown command: " + command);
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }
}
