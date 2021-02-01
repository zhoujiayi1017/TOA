using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;

namespace TestDemo
{
    internal class JHCLIB
    {
        public struct PARAMETER
        {
            public float MarkSpeed;//mark speed
            public float WorkSize; //work area
            public float RedSpeed; //red speed
            public float JumpSpeed;// jump speed
            public float JumpLocationDelay; //jump location delay
            public float JumpDistanceDelay; //jump distance delay
            public float OpenDelay;  //open laser delay
            public float CloseDelay; //close laser delay
            public float FoldDelay;  //fold point delay
            public float FinishDelay; //finish mark delay
            public PARAMETER(uint cmd)
            {
                this.MarkSpeed = 500.0f;
                this.WorkSize = 110.0f;
                this.RedSpeed = 3000.0f;
                this.JumpSpeed = 4000.0f;
                this.JumpLocationDelay = 0.0f;
                this.JumpDistanceDelay = 0.0f;
                this.OpenDelay = 0.0f;
                this.CloseDelay = 0.0f;
                this.FoldDelay = 0.0f;
                this.FinishDelay = 0.0f;
            }
        }
        public struct GALVOPARAM
        {
            public bool bXYExchange; //XY swap
            public bool bXAxisN;   // X axis negative
            public bool bYAxisN;   // Y axis negative
            private GALVOPARAM(uint cmd)
            {
                this.bXYExchange = false;
                this.bXAxisN = false;
                this.bYAxisN = false;
            }
        }
        public enum GalvanometerLocation
        {
            NotMove=0,
            GalvanometerCenter,
            SpecifiedLocation,
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct JHCStatusdef
        {
            public byte MotorStatus;
            public byte InState;
            public byte RunStatus;
            public byte LaserState;
            private JHCStatusdef(uint cmd)
            {
                this.MotorStatus = 0;
                this.InState = 0;
                this.RunStatus = 0;
                this.LaserState = 0;
            }
        };
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern bool JHCOpenDevice(); //Open USB connection, connect to Control Board, communicate with Control Board
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCCloseDevice();//Disconnect USB connection, disconnect connection with Control Board
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern bool JHCIsOpen();     //Return to the status of Connection with Control Board
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCSChSetPWM(uint Freq, byte Duty, bool Reverse);//Set Laser PWM, CO2 and Fiber Laser are both supported
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCSchOutLine(float x1, float y1, float x2, float y2);//line interpolation, coordinate of starting and ending position 
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCBufStart();//Buffer memory of initialization data      
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCCorrectionSet(float To_Jz_X, float To_Jz_Y, float Px_Jz_X, float Px_Jz_Y,
                                    float Tx_Jz_X, float Tx_Jz_Y, float Scal_X, float Scal_Y);//Set Correction Parameters
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCSchLaserOut(float power, int type, int mode);//Set Laser output and marking pattern
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCSchOutRect(float x1, float y1, float x2, float y2);//Insertion point of rectangular, coordinates of two points in diagonal line
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCSchOutCircle(float x, float y, float r);//Insertion point of circle, coordinates of the centre and radius
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCSchOutArc(float x, float y, float r, float starAngle, float endAngle);//Insertion point of arc, coordinats of the centre/radius/starting angle and ending angle
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCSchOutEllipse(float x, float y, float rx, float ry);//Insertion point of ellipse, coordinates of centre and radius of X/Y.
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCSchOutEArc(float x, float y, float rx, float ry, float starAngle, float endAngle);//Insertion point of ellipse arc, coordinates of centre/radius of X/Y, starting angle and ending angle
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCCancel();//Cancel marking, stop galvo output
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCPause();//Pause marking, pause galvo output
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCResume();//In the status of Pause Marking, back to marking, continue galvo output     
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCParameterSet(PARAMETER param);//Parameters Setting
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCGalvoParameterSet(GALVOPARAM param);//Galvo Parameters Setting   
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCStartMarking();    //Start marking, start transfer output singals from Control Board to galvo through this interface program
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern int JHCGetMarkingState();//Access to the status of marking, it is the status to send if WorkThread ends or not.
      
        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern int JHCGetSystemState(ref JHCStatusdef pStatusdef, bool bfootFlag);//Access to the status of system, including the status of auxiliary shaft/IO input/Fiber laser and marking.

        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern int JHCIoOut(byte data1, byte data2);//IO output

        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern int JHCMotorOut(byte motorType, float distance, float resolution);//Control of auxiliary shaft

        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern int JHCMotorSet(byte Id, float minSpeed, float maxSpeed, float acSpeed);//Parameter setting of auxiliary shaft

        [DllImport("JHCLIB.dll", CharSet = CharSet.Auto)]
        public static extern void JHCSchOutPoint(float x, float y, float t);//Point output/position and time of marking
    }
}
