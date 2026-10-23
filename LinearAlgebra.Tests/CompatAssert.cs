global using Assert = MyCompat.Assert;

namespace MyCompat;

public static class Assert
{
    public static void AreEqual<T>(T expected, T actual)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
        {
            throw new Xunit.Sdk.XunitException($"Expected: {expected}\nActual:   {actual}");
        }
    }

    public static void AreEqual(double expected, double actual, double precision)
    {
        if (Math.Abs(expected - actual) > precision)
        {
            throw new Xunit.Sdk.XunitException($"Expected: {expected}\nActual:   {actual}\nTolerance: {precision}");
        }
    }

    public static void IsGreaterThanOrEqualTo(double lowerBound, double actual)
    {
        if (!(actual >= lowerBound))
        {
            throw new Xunit.Sdk.XunitException($"Expected value >= {lowerBound}, but was {actual}.");
        }
    }

    public static void IsLessThanOrEqualTo(double upperBound, double actual)
    {
        if (!(actual <= upperBound))
        {
            throw new Xunit.Sdk.XunitException($"Expected value <= {upperBound}, but was {actual}.");
        }
    }

    public static void IsTrue(bool condition)
    {
        if (!condition)
        {
            throw new Xunit.Sdk.XunitException("Expected condition to be true.");
        }
    }
}
