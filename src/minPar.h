#pragma once
#ifdef _WIN32
#  define EXPORT __declspec(dllexport) 
#else
#  define EXPORT
#  define CALL
#endif

/**
 * @brief Compute the minimum distance between a point given by x,y and the path
 *        given by the set of points {pathx, pathy}.
 *        Distances will be weighted by the given sigma values
 * 
 * @param x Point x position
 * @param y Point y position
 * @param pathx set of x coordinates of path
 * @param pathy set of y cooridnates of path
 * @param sigmax error on x position
 * @param sigmay error on y position
 * @param M number of points in the path
 * @return double the minimum distance between point and path
 */
EXPORT double minDist(double x, double y, 
               double* pathx, double* pathy,
               double sigmax,double sigmay, 
               int M);
/**
 * @brief Given a set of points defined by {xs, ys}, compute each minimum 
 *        distance to the path given by {pathx, pathy}, see minDist. Output
 *        is put into array supplied to function.
 * 
 * @param xs Set of x coordinates of points
 * @param ys Set of y coordinated of points
 * @param pathx Set of x coordinates of path
 * @param pathy Set of y coordinates of path
 * @param out Array of same length of xs to put results in
 * @param sigmax Error on x point positions, taken to be constant
 * @param sigmay Error on y point positions, taken to be constant
 * @param N Number of points
 * @param M Length of path
 */
EXPORT int minDists(double* xs, double* ys, 
                double* pathx, double* pathy, double* out,
                double sigmax,double sigmay, 
                int N, int M);

/**
 * @brief Given a point (x,y), and a path {pathx, pathy} parametrized by lengths,
 *        computes the length along the path at which the distance to the point
 *        is minimized.
 * 
 * @param x Point x position
 * @param y Point y position
 * @param pathx set of x coordinates of path
 * @param pathy set of y cooridnates of path
 * @param lengths set of parametrization lengths
 * @param sigmax error on x position
 * @param sigmay error on y position
 * @param M number of points in the path
 * @return double The length that minimizes the distance
 */
EXPORT double minLength(double x, double y, 
                 double* pathx, double* pathy, double* lengths,
                 double sigmax,double sigmay, 
                 int M);

/**
 * @brief Given a set of point {x,y}, and a path {pathx, pathy} parametrized by lengths,
 *        computes the length along the path at which the distance to the point
 *        is minimized for each point, see minLength. Output
 *        is put into array supplied to function.
 * 
 * @param xs Set of x coordinates of points
 * @param ys Set of y coordinated of points
 * @param pathx Set of x coordinates of path
 * @param pathy Set of y coordinates of path
 * @param lengths Set of parametrization lengths
 * @param out Array of same length of xs to put results in
 * @param sigmax Error on x point positions, taken to be constant
 * @param sigmay Error on y point positions, taken to be constant
 * @param N Number of points
 * @param M Length of path
 */
EXPORT int minLengths(double* xs, double* ys, 
                   double* pathx, double* pathy, double* lengths, double* out,
                   double sigmax,double sigmay, 
                   int N, int M);

/**
 * @brief Returns the lower and upper index bounds of the array length such that
 *        each length is within +/- limit of prev.
 * 
 * @param lengths Set of length values
 * @param M Number of lengths
 * @param prev The central limiting value
 * @param limit How far to deviate in either direction of prev
 * @param range Output array to put the lower and upper index.
 */
EXPORT void getRange(double* lengths, int M, double prev, double limit, int* range);

/**
 * @brief Given a point (x,y), and a path {pathx, pathy} parametrized by lengths,
 *        computes the length along the path, with hystersis, 
 *        at which the distance to the point is minimized. Will only look at 
 *        lengths within +/- limit of prev. 
 * 
 * @param x Point x position
 * @param y Point y position
 * @param pathx set of x coordinates of path
 * @param pathy set of y cooridnates of path
 * @param lengths set of parametrization lengths
 * @param sigmax error on x position
 * @param sigmay error on y position
 * @param M number of points in the path
 * @param prev central limiting value
 * @param limit radius around prev to look at 
 * @return double The length that minimizes the distance
 */
EXPORT double minLengthHist(double x, double y, 
                 double* pathx, double* pathy, double* lengths,
                 double sigmax,double sigmay, 
                 int M, 
                 double prev, double limit);

/**
 * @brief Same as minLengthHist, but over a set of points {x,y}. First value in 
 *        array is compued without hysteresis to give starting point. Results are
 *        put into output array which needs to be supplied to function.
 * 
 * @param xs Set of x coordinates of points
 * @param ys Set of y coordinated of points
 * @param pathx Set of x coordinates of path
 * @param pathy Set of y coordinates of path
 * @param lengths Set of parametrization lengths
 * @param out Array of same length of xs to put results in
 * @param sigmax Error on x point positions, taken to be constant
 * @param sigmay Error on y point positions, taken to be constant
 * @param N Number of points
 * @param M Length of path
 * @param limit radius around each previous length to look at.
 */
EXPORT int minLengthsHist(double* x, double* y, 
                      double* pathx, double* pathy, double* lengths, 
                      double* output,
                      double sigmax,double sigmay, 
                      int N, int M, double limit);
