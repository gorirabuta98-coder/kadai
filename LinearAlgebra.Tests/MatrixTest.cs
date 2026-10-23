namespace LinearAlgebra.Tests;

using static LinearAlgebra.Matrix;

public class MatrixTest
{
    readonly Random rand = new();

    [Fact]
    public void TestEye()
    {
        int n = 5;
        double[,] A = { { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1 } };

        var I = Eye(n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(A[i, j], I[i, j]);
            }
        }
    }

    [Fact]
    public void TestOnes()
    {
        int m = 5;
        int n = 3;
        double[,] A = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };

        var B = Ones(m, n);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(A[i, j], B[i, j]);
            }
        }

        double[] a = [1, 1, 1, 1, 1];

        var b = Ones(m);

        for (int i = 0; i < m; i++)
            Assert.AreEqual(a[i], b[i]);
    }

    [Fact]
    public void TestUniform()
    {
        int p = 10000;
        int m = 5;
        int n = 3;
        var S0 = new double[m, n];
        var S1 = new double[m, n];
        var S2 = new double[m, n];
        var S3 = new double[m, n];

        for (int i = 0; i < p; i++)
        {
            var X = Uniform(m, n, rand);
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    S0[j, k] = Math.Min(S0[j, k], X[j, k]);
                    S1[j, k] += X[j, k];
                    S2[j, k] += X[j, k] * X[j, k];
                    S3[j, k] = Math.Max(S3[j, k], X[j, k]);
                }
            }
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.IsGreaterThanOrEqualTo(0, S0[i, j]);
                Assert.AreEqual(1 / 2.0, S1[i, j]/(double)p, 1e-1);
                Assert.AreEqual(1 / 3.0, S2[i, j]/(double)p, 1e-1);
                Assert.IsLessThanOrEqualTo(1, S3[i, j]);
            }
        }

        var s0 = new double[m];
        var s1 = new double[m];
        var s2 = new double[m];
        var s3 = new double[m];

        for (int i = 0; i < p; i++)
        {
            var x = Uniform(m, rand);
            for (int j = 0; j < m; j++)
            {
                s0[j] = Math.Min(s0[j], x[j]);
                s1[j] += x[j];
                s2[j] += x[j] * x[j];
                s3[j] = Math.Max(s3[j], x[j]);
            }
        }

        for (int i = 0; i < m; i++)
        {
            Assert.IsGreaterThanOrEqualTo(0, s0[i]);
            Assert.AreEqual(1 / 2.0, s1[i]/(double)p, 1e-1);
            Assert.AreEqual(1 / 3.0, s2[i]/(double)p, 1e-1);
            Assert.IsLessThanOrEqualTo(1, s3[i]);
        }
    }

    [Fact]
    public void TestNormal()
    {
        int p = 10000;
        int m = 5;
        int n = 3;
        var S1 = new double[m, n];
        var S2 = new double[m, n];

        for (int i = 0; i < p; i++)
        {
            double[,] X = Normal(m, n, rand);
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    S1[j, k] += X[j, k];
                    S2[j, k] += X[j, k] * X[j, k];
                }
            }
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(0.0, S1[i, j]/(double)p, 1e-1);
                Assert.AreEqual(1.0, S2[i, j]/(double)p, 1e-1);
            }
        }

        var s1 = new double[m];
        var s2 = new double[m];

        for (int i = 0; i < p; i++)
        {
            var x = Normal(m, rand);
            for (int j = 0; j < m; j++)
            {
                s1[j] += x[j];
                s2[j] += x[j] * x[j];
            }
        }

        for (int i = 0; i < m; i++)
        {
            Assert.AreEqual(0.0, s1[i]/(double)p, 1e-1);
            Assert.AreEqual(1.0, s2[i]/(double)p, 1e-1);
        }
    }

    [Fact]
    public void TestCopy()
    {
        int m = 5;
        int n = 3;
        double[,] X = { { 0, 0.6, 0.1 }, { 0.7, 0.6, 0.2 }, { 0.5, 0.8, 0.8 }, { 0.4, 0.8, 0 }, { 0.9, 0.5, 0.4 } };
        double[,] XX = { { 0, 0.6, 0.1 }, { 0.7, 0.6, 0.2 }, { 0.5, 0.8, 0.8 }, { 0.4, 0.8, 0 }, { 0.9, 0.5, 0.4 } };

        var Y = Copy(X);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], Y[i, j]);
                Assert.AreEqual(XX[i, j], X[i, j]);
            }
        }

        Y = new double[m,n];
        Copy(X, Y);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], Y[i, j]);
                Assert.AreEqual(XX[i, j], X[i, j]);
            }
        }

        double[] x = [0.1, 0.9, 0.7, 0.5, 0.4];
        double[] xx = [0.1, 0.9, 0.7, 0.5, 0.4];

        var y = Copy(x);

        for (int i = 0; i < m; i++)
        {    
            Assert.AreEqual(xx[i], y[i]);
            Assert.AreEqual(xx[i], x[i]);
        }

        y = new double[m];
        Copy(x, y);

        for (int i = 0; i < m; i++)
        {    
            Assert.AreEqual(xx[i], y[i]);
            Assert.AreEqual(xx[i], x[i]);
        }
    }

    [Fact]
    public void TestTranspose()
    {
        int m = 5;
        int n = 3;
        double[,] X = { { 0, 0.6, 0.4 }, { 0.8, 0.5, 0.4 }, { 0.8, 0.9, 0.8 }, { 0.7, 0.6, 0 }, { 0.1, 0.8, 0.1 } };
        double[,] XX = { { 0, 0.6, 0.4 }, { 0.8, 0.5, 0.4 }, { 0.8, 0.9, 0.8 }, { 0.7, 0.6, 0 }, { 0.1, 0.8, 0.1 } };
        double[,] YY = { { 0, 0.8, 0.8, 0.7, 0.1 }, { 0.6, 0.5, 0.9, 0.6, 0.8 }, { 0.4, 0.4, 0.8, 0, 0.1 } };

        var Y = Transpose(X);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(YY[j, i], Y[j, i]);
                Assert.AreEqual(XX[i, j], X[i, j]);
            }
        }
    }

    [Fact]
    public void TestScalar()
    {
        int m = 5;
        int n = 3;
        double[,] X = { { 0.1, 0.3, 0.2 }, { 0.3, 0.5, 0.4 }, { 0.8, 0.4, 0 }, { 0.8, 0.6, 0.9 }, { 0, 0.6, 0.1 } };
        double[,] XX = { { 0.1, 0.3, 0.2 }, { 0.3, 0.5, 0.4 }, { 0.8, 0.4, 0 }, { 0.8, 0.6, 0.9 }, { 0, 0.6, 0.1 } };
        double[,] YY = { { 0.3, 0.9, 0.6 }, { 0.9, 1.5, 1.2 }, { 2.4, 1.2, 0 }, { 2.4, 1.8, 2.7 }, { 0, 1.8, 0.3 } };
        double alpha = 3;

        var Y = Scalar(X, alpha);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j]);
                Assert.AreEqual(YY[i, j], Y[i, j], 1e-12);
            }
        }

        double[] x = [0.5, 0.4, 0.7, 0.7, 0.5];
        double[] xx = [0.5, 0.4, 0.7, 0.7, 0.5];
        double[] yy = [3.5, 2.8, 4.9, 4.9, 3.5];
        alpha = 7;

        var y = Scalar(x, alpha);

        for (int i = 0; i < m; i++)
        {
            Assert.AreEqual(xx[i], x[i]);
            Assert.AreEqual(yy[i], y[i], 1e-12);
        }
    }

    [Fact]
    public void TestAdd()
    {
        int m = 3;
        int n = 2;
        double[,] X = { { 1.8, 2.7 }, { 0, 1.2 }, { 1.2, 1.2 } };
        double[,] Y = { { 0.4, 0.3 }, { 0.4, -0.6 }, { 0.5, 0.3 } };
        double[,] XX = { { 1.8, 2.7 }, { 0, 1.2 }, { 1.2, 1.2 } };
        double[,] YY = { { 0.4, 0.3 }, { 0.4, -0.6 }, { 0.5, 0.3 } };
        double[,] ZZ = { { 2.2, 3.0 }, { 0.4, 0.6 }, { 1.7, 1.5 } };

        var Z = Add(X, Y);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j]);
                Assert.AreEqual(YY[i, j], Y[i, j]);
                Assert.AreEqual(ZZ[i, j], Z[i, j], 1e-12);
            }
        }

        double a = 1.5;
        ZZ = new double[,]{ { 3.3, 4.2 }, { 1.5, 2.7 }, { 2.7, 2.7 } };

        Z = Add(X, a);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j]);
                Assert.AreEqual(ZZ[i, j], Z[i, j], 1e-12);
            }
        }

        n = 5;
        double[] x = [2.0, 1.0, 2.5, 0.5, 4.0];
        double[] y = [0.2, 0.5, 0.8, 0.3, 0.1];
        double[] xx = [2.0, 1.0, 2.5, 0.5, 4.0];
        double[] yy = [0.2, 0.5, 0.8, 0.3, 0.1];
        double[] zz = [2.2, 1.5, 3.3, 0.8, 4.1];

        var z = Add(x, y);

        for (int i = 0; i < n; i++)
        {
            Assert.AreEqual(xx[i], x[i]);
            Assert.AreEqual(yy[i], y[i]);
            Assert.AreEqual(zz[i], z[i], 1e-12);
        }

        a = -0.2;
        zz = [1.8, 0.8, 2.3, 0.3, 3.8];

        z = Add(x, a);

        for (int i = 0; i < n; i++)
        {
            Assert.AreEqual(xx[i], x[i]);
            Assert.AreEqual(zz[i], z[i], 1e-12);
        }
    }

    [Fact]
    public void TestCwiseMultiply()
    {
        int m = 3;
        int n = 2;
        double[,] X = { { 1.8, 2.7 }, { 0, 1.2 }, { 1.2, 1.2 } };
        double[,] Y = { { 0.4, 0.3 }, { 0.4, -0.6 }, { 0.5, 0.3 } };
        double[,] XX = { { 1.8, 2.7 }, { 0, 1.2 }, { 1.2, 1.2 } };
        double[,] YY = { { 0.4, 0.3 }, { 0.4, -0.6 }, { 0.5, 0.3 } };
        double[,] ZZ = { { 0.72, 0.81 }, { 0, -0.72 }, { 0.6, 0.36 } };

        var Z = CwiseMultiply(X, Y);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j]);
                Assert.AreEqual(YY[i, j], Y[i, j]);
                Assert.AreEqual(ZZ[i, j], Z[i, j], 1e-12);
            }
        }

        n = 5;
        double[] x = [2.0, 1.0, 2.5, 0.5, 4.0];
        double[] y = [0.2, 0.5, 0.8, 0.3, 0.1];
        double[] xx = [2.0, 1.0, 2.5, 0.5, 4.0];
        double[] yy = [0.2, 0.5, 0.8, 0.3, 0.1];
        double[] zz = [0.4, 0.5, 2.0, 0.15, 0.4];

        var z = CwiseMultiply(x, y);

        for (int i = 0; i < n; i++)
        {
            Assert.AreEqual(xx[i], x[i]);
            Assert.AreEqual(yy[i], y[i]);
            Assert.AreEqual(zz[i], z[i], 1e-12);
        }
    }

    [Fact]
    public void TestCwiseDivide()
    {
       int m = 3;
        int n = 2;
        double[,] X = { { 1.8, 2.7 }, { 0, 1.2 }, { 1.2, 1.2 } };
        double[,] Y = { { 0.4, 0.3 }, { 0.4, -0.6 }, { 0.5, 0.3 } };
        double[,] XX = { { 1.8, 2.7 }, { 0, 1.2 }, { 1.2, 1.2 } };
        double[,] YY = { { 0.4, 0.3 }, { 0.4, -0.6 }, { 0.5, 0.3 } };
        double[,] ZZ = { { 4.5, 9.0 }, { 0, -2.0 }, { 2.4, 4.0 } };

        var Z = CwiseDivide(X, Y);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j]);
                Assert.AreEqual(YY[i, j], Y[i, j]);
                Assert.AreEqual(ZZ[i, j], Z[i, j], 1e-12);
            }
        }

        n = 5;
        double[] x = [2.0, 1.0, 2.5, 0.5, 4.0];
        double[] y = [0.2, 0.5, 0.8, 0.2, 0.1];
        double[] xx = [2.0, 1.0, 2.5, 0.5, 4.0];
        double[] yy = [0.2, 0.5, 0.8, 0.2, 0.1];
        double[] zz = [10, 2, 3.125, 2.5, 40];

        var z = CwiseDivide(x, y);

        for (int i = 0; i < n; i++)
        {
            Assert.AreEqual(xx[i], x[i]);
            Assert.AreEqual(yy[i], y[i]);
            Assert.AreEqual(zz[i], z[i], 1e-12);
        }
    }

    [Fact]
    public void TestMultiply()
    {
        int m = 3;
        int n = 2;
        int l = 4;
        double[,] X = { { 0.6, 0.9, 0.6, 0.2 }, { 0.8, 0.4, 0.9, 0.6 }, { 0.1, 0.6, 0.1, 0 } };
        double[,] Y = { { 0.2, 0.7 }, { 0, 0.2 }, { 0.7, 0.4 }, { 0.9, 0.7 } };
        double[,] XX = { { 0.6, 0.9, 0.6, 0.2 }, { 0.8, 0.4, 0.9, 0.6 }, { 0.1, 0.6, 0.1, 0 } };
        double[,] YY = { { 0.2, 0.7 }, { 0, 0.2 }, { 0.7, 0.4 }, { 0.9, 0.7 } };
        double[,] ZZ = { { 0.72, 0.98 }, { 1.33, 1.42 }, { 0.09, 0.23 } };

        double[,] Z = Multiply(X, Y);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < l; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j]);
            }
        }
        for (int i = 0; i < l; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(YY[i, j], Y[i, j]);
            }
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(ZZ[i, j], Z[i, j], 1e-12);
            }
        }

        m = 3;
        n = 4;
        X = new double[,] { { 0.8, 0.2, 0.1, 0.6 }, { 0, 0.6, 0.2, 0.2 }, { 0.3, 0.9, 0.8, 0.2 } };
        double[] y = [0.1, 0.5, 0.2, 0.1];
        XX = new double[,] { { 0.8, 0.2, 0.1, 0.6 }, { 0, 0.6, 0.2, 0.2 }, { 0.3, 0.9, 0.8, 0.2 } };
        double[] yy = [0.1, 0.5, 0.2, 0.1];
        double[] zz = [0.26, 0.36, 0.66];

        double[] z = Multiply(X, y);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j]);
            }
        }
        for (int i = 0; i < n; i++)
        {
            Assert.AreEqual(yy[i], y[i]);
        }
        for (int i = 0; i < m; i++)
        {
            Assert.AreEqual(zz[i], z[i], 1e-12);
        }

        m = 4;
        n = 3;
        X = new double[,] { { 0, 0.1, 0.7, 0.3 }, { 0.6, 0.5, 0.4, 0.3 }, { 0.5, 0, 0.9, 0.1 } };
        y = [0.4, 0.5, 0.8];
        XX = new double[,] { { 0, 0.1, 0.7, 0.3 }, { 0.6, 0.5, 0.4, 0.3 }, { 0.5, 0, 0.9, 0.1 } };
        yy = [0.4, 0.5, 0.8];
        zz = [0.7, 0.29, 1.2, 0.35];

        z = Multiply(y, X);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j]);
            }
        }
        for (int i = 0; i < n; i++)
        {
            Assert.AreEqual(yy[i], y[i]);
        }
        for (int i = 0; i < m; i++)
        {
            Assert.AreEqual(zz[i], z[i], 1e-12);
        }
    }

    [Fact]
    public void TestInner()
    {
        int n = 5;
        double[] x = [0.5, 0.3, 0.8, 0.9, 0.1];
        double[] y = [0.2, 0.6, 0.5, 0, 0.5];
        double[] xx = [0.5, 0.3, 0.8, 0.9, 0.1];
        double[] yy = [0.2, 0.6, 0.5, 0, 0.5];
        double zz = 0.73;

        double z = Inner(x, y);

        for (int i = 0; i < n; i++)
        {
            Assert.AreEqual(xx[i], x[i]);
        }
        for (int i = 0; i < n; i++)
        {
            Assert.AreEqual(yy[i], y[i]);
        }
        Assert.AreEqual(zz, z, 1e-12);
    }

    [Fact]
    public void TestOuter()
    {
        int m = 3;
        int n = 2;
        double[] x = [0.9, 0.3, 0.8];
        double[] y = [0.8, 0.5];
        double[] xx = [0.9, 0.3, 0.8];
        double[] yy = [0.8, 0.5];
        double[,] ZZ = { { 0.72, 0.45 }, { 0.24, 0.15 }, { 0.64, 0.4 } };

        double[,] Z = Outer(x, y);

        for (int i = 0; i < m; i++)
        {
            Assert.AreEqual(xx[i], x[i]);
        }
        for (int i = 0; i < n; i++)
        {
            Assert.AreEqual(yy[i], y[i]);
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.AreEqual(ZZ[i, j], Z[i, j], 1e-12);
            }
        }
    }

    [Fact]
    public void TestGetColumn()
    {
        double[,] X = {
            {0.53,1.30,1.34,0.20,0.67,1.03,0.88},
            {1.83,0.43,3.03,0.12,1.20,0.72,1.14},
            {2.25,0.34,0.72,1.48,0.71,0.30,1.06},
            {0.86,3.57,0.06,1.40,1.63,0.29,0.80},
            {0.31,2.76,0.71,1.41,0.48,0.78,2.94}
            };
        double[] yy = [1.20,0.71,1.63];
        int col = 4;
        int pos = 1;
        int len = 3;

        double[] y = GetColumn(X, col, pos, len);

        for (int i = 0; i < len; i++)
        {
            Assert.AreEqual(yy[i], y[i], 1e-12);
        }
    }

    [Fact]
    public void TestSetColumn()
    {
        double[,] X = {
            {0.53,1.30,1.34,0.20,0.67,1.03,0.88},
            {1.83,0.43,3.03,0.12,1.20,0.72,1.14},
            {2.25,0.34,0.72,1.48,0.71,0.30,1.06},
            {0.86,3.57,0.06,1.40,1.63,0.29,0.80},
            {0.31,2.76,0.71,1.41,0.48,0.78,2.94}
            };
        double[,] XX = {
            {0.53,1.30,1.34,0.20,0.67,1.03,0.88},
            {1.83,0.43,1.02,0.12,1.20,0.72,1.14},
            {2.25,0.34,0.33,1.48,0.71,0.30,1.06},
            {0.86,3.57,0.91,1.40,1.63,0.29,0.80},
            {0.31,2.76,0.61,1.41,0.48,0.78,2.94}
            };
        double[] y = [1.02,0.33,0.91,0.61];
        int col = 2;
        int pos = 1;

        SetColumn(X, col, pos, y);

        int m = X.GetLength(0);
        int n = X.GetLength(1);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j], 1e-12);
            }
        }
    }

    [Fact]
    public void TestGetRow()
    {
        double[,] X = {
            {0.53,1.30,1.34,0.20,0.67,1.03,0.88},
            {1.83,0.43,3.03,0.12,1.20,0.72,1.14},
            {2.25,0.34,0.72,1.48,0.71,0.30,1.06},
            {0.86,3.57,0.06,1.40,1.63,0.29,0.80},
            {0.31,2.76,0.71,1.41,0.48,0.78,2.94}
            };
        double[] yy = [0.31,2.76,0.71,1.41];
        int row = 4;
        int pos = 0;
        int len = 4;

        double[] y = GetRow(X, row, pos, len);

        for (int i = 0; i < len; i++)
        {
            Assert.AreEqual(yy[i], y[i], 1e-12);
        }
    }

    [Fact]
    public void TestSetRow()
    {
        double[,] X = {
            {0.53,1.30,1.34,0.20,0.67,1.03,0.88},
            {1.83,0.43,3.03,0.12,1.20,0.72,1.14},
            {2.25,0.34,0.72,1.48,0.71,0.30,1.06},
            {0.86,3.57,0.06,1.40,1.63,0.29,0.80},
            {0.31,2.76,0.71,1.41,0.48,0.78,2.94}
            };
        double[,] XX = {
            {0.53,1.30,1.34,0.20,0.67,1.03,0.88},
            {1.83,0.43,0.53,0.88,0.61,0.01,1.14},
            {2.25,0.34,0.72,1.48,0.71,0.30,1.06},
            {0.86,3.57,0.06,1.40,1.63,0.29,0.80},
            {0.31,2.76,0.71,1.41,0.48,0.78,2.94}
            };
        double[] y = [0.53,0.88,0.61,0.01];
        int col = 1;
        int pos = 2;

        SetRow(X, col, pos, y);

        int m = X.GetLength(0);
        int n = X.GetLength(1);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j], 1e-12);
            }
        }
    }

    [Fact]
    public void TestGetSubvector()
    {
        double[] x = [0.32,0.31,0.86,0.03,0.16,0.62,1.09,1.10,0.86];
        double[] yy = [0.62,1.09,1.10,0.86];
        int pos = 5;
        int len = 4;

        double[] y = GetSubvector(x, pos, len);

        for (int i = 0; i < len; i++)
        {
            Assert.AreEqual(yy[i], y[i], 1e-12);
        }
    }

    [Fact]
    public void TestSetSubvector()
    {
        double[] x = [0.32,0.31,0.86,0.03,0.16,0.62,1.09,1.10,0.86];
        double[] xx = [0.32,0.02,0.45,0.89,0.23,0.56,1.09,1.10,0.86];
        double[] y = [0.02,0.45,0.89,0.23,0.56];
        int pos = 1;

        SetSubvector(x, pos, y);

        int n = x.Length;
        for(int i = 0; i < n; i++)
        {
            Assert.AreEqual(xx[i], x[i], 1e-12);
        }
    }

    [Fact]
    public void TestCwiseFunction()
    {
        double[,] X = {
            {1.10,1.54,0.09},
            {1.49,0.74,1.06},
            {2.35,1.29,0.89}};
        double[,] XX = {
            {1.10,1.54,0.09},
            {1.49,0.74,1.06},
            {2.35,1.29,0.89}};
        double[,] YY = {
            {0.4536,0.0308,0.9960},
            {0.0807,0.7385,0.4889},
            {-0.7027,0.2771,0.6294}};

        double[,] Y = CwiseFunction(X, Math.Cos);

        int m = X.GetLength(0);
        int n = X.GetLength(1);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                Assert.AreEqual(XX[i, j], X[i, j], 1e-12);
                Assert.AreEqual(YY[i, j], Y[i, j], 1e-4);
            }
        }

        double[] x = [1.10,1.54,0.09,1.49,0.74,1.06,2.35];
        double[] xx = [1.10,1.54,0.09,1.49,0.74,1.06,2.35];
        double[] yy = [0.8912,0.9995,0.0899,0.9967,0.6743,0.8724,0.7115];

        double[] y = CwiseFunction(x, Math.Sin);

        for (int i = 0; i < x.Length; i++)
        {
            Assert.AreEqual(xx[i], x[i], 1e-12);
            Assert.AreEqual(yy[i], y[i], 1e-4);
        }
    }
}