/**
 * Camera Stabilizer using OpenCV.js
 * Computes an affine transformation matrix to compensate for camera jitter/pan.
 */

export interface Matrix {
  m00: number; m01: number; m02: number;
  m10: number; m11: number; m12: number;
}

export class CameraStabilizer {
  private refGray: any = null;
  private prevGray: any = null;
  private isCcReady = false;

  constructor() {
    // Check if cv is available (loaded via CDN)
    if (typeof (window as any).cv !== 'undefined') {
      this.isCcReady = true;
    }
  }

  /**
   * Set the reference frame (taken during Setup phase).
   * All future frames will be stabilized relative to this.
   */
  setReferenceFrame(video: HTMLVideoElement | HTMLCanvasElement | ImageBitmap) {
    const cv = (window as any).cv;
    if (!cv || !cv.Mat) return;

    if (this.refGray) this.refGray.delete();
    
    // Create Mat from source
    const src = cv.imread(video);
    this.refGray = new cv.Mat();
    cv.cvtColor(src, this.refGray, cv.COLOR_RGBA2GRAY);
    src.delete();
    
    console.log("📸 Stabilizer: Reference frame set.");
  }

  /**
   * Computes the transformation matrix from the reference to the current frame.
   */
  computeTransform(currentVideo: HTMLVideoElement | HTMLCanvasElement): Matrix | null {
    const cv = (window as any).cv;
    if (!cv || !this.refGray) return null;

    try {
      const src = cv.imread(currentVideo);
      const currGray = new cv.Mat();
      cv.cvtColor(src, currGray, cv.COLOR_RGBA2GRAY);

      // We use estimateAffinePartial2D which handles translation, rotation, and uniform scale
      // Ideal for handheld shake or simple pan/tilt
      
      // For performance in JS we'll use ORB feature matching or simple ECC
      // For now, let's use a simpler approach: ORB keypoints
      const orb = new cv.ORB();
      const kp1 = new cv.KeyPointVector();
      const des1 = new cv.Mat();
      const kp2 = new cv.KeyPointVector();
      const des2 = new cv.Mat();

      orb.detectAndCompute(this.refGray, new cv.Mat(), kp1, des1);
      orb.detectAndCompute(currGray, new cv.Mat(), kp2, des2);

      const bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
      const matches = new cv.DMatchVector();
      bf.match(des1, des2, matches);

      // Filter good matches
      const goodMatches: any[] = [];
      for (let i = 0; i < matches.size(); i++) {
        const m = matches.get(i);
        if (m.distance < 30) goodMatches.push(m);
      }

      if (goodMatches.length < 10) {
        src.delete(); currGray.delete(); orb.delete(); 
        kp1.delete(); des1.delete(); kp2.delete(); des2.delete(); 
        bf.delete(); matches.delete();
        return null; 
      }

      const pts1 = [];
      const pts2 = [];
      for (const m of goodMatches) {
        pts1.push(kp1.get(m.queryIdx).pt.x);
        pts1.push(kp1.get(m.queryIdx).pt.y);
        pts2.push(kp2.get(m.trainIdx).pt.x);
        pts2.push(kp2.get(m.trainIdx).pt.y);
      }

      const mat1 = cv.matFromArray(pts1.length / 2, 1, cv.CV_32FC2, pts1);
      const mat2 = cv.matFromArray(pts2.length / 2, 1, cv.CV_32FC2, pts2);

      const M = cv.estimateAffinePartial2D(mat1, mat2);

      let result: Matrix | null = null;
      if (!M.empty()) {
        result = {
          m00: M.data64F[0], m01: M.data64F[1], m02: M.data64F[2],
          m10: M.data64F[3], m11: M.data64F[4], m12: M.data64F[5]
        };
      }

      // Cleanup
      src.delete(); currGray.delete(); orb.delete(); 
      kp1.delete(); des1.delete(); kp2.delete(); des2.delete(); 
      bf.delete(); matches.delete(); mat1.delete(); mat2.delete(); M.delete();

      return result;
    } catch (e) {
      console.warn("Stabilization failed:", e);
      return null;
    }
  }

  /**
   * Apply transformation to a point
   */
  static transformPoint(x: number, y: number, m: Matrix): [number, number] {
    return [
      m.m00 * x + m.m01 * y + m.m02,
      m.m10 * x + m.m11 * y + m.m12
    ];
  }
}
