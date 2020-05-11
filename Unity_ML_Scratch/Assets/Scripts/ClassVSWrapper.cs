using System.Runtime.InteropServices;

public static class ClassVSWrapper
{
    [DllImport("ML_project")]
    public static extern int my_add(int x, int y);
    [DllImport("ML_project")]
    public static extern int my_mul(int x, int y);
}