using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO.Pipes;
using System.Diagnostics;
using System.Threading.Tasks;

namespace TestDemo
{
    class PipeServerManager : IDisposable
    {
        #region Member

        public enum CameraTypeEnum { RealSense, WebCamera }

        // 推論結果の値
        private CameraTypeEnum _cameraType;

        // 重複呼び出し検出用
        private bool disposedValue = false;

        // ログ出力用
        private static log4net.ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

        // 起動するEXE
        private static readonly string EXE_NAME = "python";
        //private static readonly string EXE_NAME = @"C:\Users\s_tanimoto\AppData\Local\Programs\Python\Python36\python.exe";

        // パイプ名（Pythonと同じ名前にする必要がある）
        private static readonly string PIPE_NAME = "pipe-throat";

        // 名前付きパイプ
        private NamedPipeClientStream _namedPipeClient = null;

        private bool _isConnected = false;

        #endregion

        #region コンストラクタ

        public PipeServerManager()
        {
            this._cameraType = CameraTypeEnum.WebCamera;
        }

        #endregion

        #region プロパティ

        /// <summary>
        /// 起動して接続済みか否か
        /// </summary>
        public bool IsConnected
        {
            get { return this._isConnected; }
            set { this._isConnected = value; }
        }

        #endregion

        #region Public関数

        /// <summary>
        /// サーバープロセス起動
        /// </summary>
        /// <param name="cameraType">利用するカメラの種別</param>
        public void WakeupServer(CameraTypeEnum cameraType)
        {
            // カメラ種別を決定
            this._cameraType = cameraType;
            // 起動
            this.StartProcess();
            // パイプ接続
            this.CreateAndConnectPipe();
            // 接続して起動済みに設定
            this.IsConnected = true;
        }

        /// <summary>
        /// 名前付きパイプに書き込み
        /// </summary>
        /// <param name="data">書き込む値</param>
        public void WriteToServer(byte[] data)
        {
            logger.Debug(data);
            // 名前付きパイプに書き込み
            this._namedPipeClient.Write(data, 0, data.Length);
            // TODO: 必要？
            this._namedPipeClient.WaitForPipeDrain();
        }

        /// <summary>
        /// 名前付きパイプに書き込み
        /// </summary>
        /// <param name="data">書き込む値</param>
        public void WriteToServer(short[] data)
        {
            //logger.Debug(data);

            byte[] outData = new byte[data.Length * 2];
            for (int i = 0; i < data.Length; i++)
            {
                byte[] values = BitConverter.GetBytes(data[i]);
                outData[i * 2] = values[0];
                outData[i * 2 + 1] = values[1];
            }

            // 名前付きパイプに書き込み
            this._namedPipeClient.Write(outData, 0, outData.Length);
            // TODO: 必要？
            this._namedPipeClient.WaitForPipeDrain();
        }

        /// <summary>
        /// 名前付きパイプから読み込み
        /// </summary>
        /// <param name="byteLength">バイト数</param>
        /// <returns>読み込んだ値</returns>
        public byte[] ReadFromServer(int byteLength)
        {
            // バッファ初期化 
            var outputByte = new byte[byteLength];
            // 名前付きパイプからバイナリ読み込み
            this._namedPipeClient.Read(outputByte, 0, outputByte.Length);

            return outputByte;
        }

        #endregion

        #region Private関数

        /// <summary>
        /// プロセス起動
        /// </summary>
        private void StartProcess()
        {
            // 起動するスクリプトを決定
            string pythonScript = "";
            if (this._cameraType.Equals(CameraTypeEnum.RealSense))
            {
                pythonScript = @"YoloThroat\realsense.py";
            } else
            {
                pythonScript = @"YoloThroat\webcam.py";
            }

            ProcessStartInfo psInfo = new ProcessStartInfo();

            psInfo.FileName = EXE_NAME;             // 実行するEXE
            psInfo.CreateNoWindow = true;           // コンソール・ウィンドウを開かない
            psInfo.UseShellExecute = false;         // シェル機能を使用しない
            psInfo.RedirectStandardOutput = true;   // 標準出力をリダイレクト
            psInfo.RedirectStandardInput = true;    // 標準入力をリダイレクト
            psInfo.Arguments = pythonScript;        // パラメータを指定

            Process process = new Process();
            process.StartInfo = psInfo;

            // 起動
            process.Start();
        }

        /// <summary>
        /// 名前付きパイプ作成して接続
        /// </summary>
        private void CreateAndConnectPipe()
        {
            this._namedPipeClient = new NamedPipeClientStream(PIPE_NAME);

            // Connect to the pipe or wait until the pipe is available.
            this._namedPipeClient.Connect();
            //Console.WriteLine("接続完了");
            logger.Info("接続完了");
        }

        /// <summary>
        /// プロセス終了
        /// </summary>
        private void ShutdownServer()
        {
            if (this._namedPipeClient != null)
            {
                if (this._namedPipeClient.IsConnected && _namedPipeClient.CanWrite)
                {
                    // 下記はサポートされてないからNG
                    //_namedPipeClient.WriteTimeout = 500;
                    // 
                    // 終了コマンドを送信
                    string finishText = new string('@', 10);
                    byte[] finishData = Encoding.ASCII.GetBytes(finishText);
                    this._namedPipeClient.Write(finishData, 0, finishData.Length);
                }
                this._namedPipeClient.Close();
                this._namedPipeClient.Dispose();
            }

            // プロセス終了
            Process[] ps = Process.GetProcessesByName(EXE_NAME);
            foreach (Process p in ps)
            {
                p.Close();
            }
        }

        #endregion

        #region IDisposable Support

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // マネージド状態を破棄します (マネージド オブジェクト)。
                    ShutdownServer();
                }

                // TODO: アンマネージド リソース (アンマネージド オブジェクト) を解放し、下のファイナライザーをオーバーライドします。
                // TODO: 大きなフィールドを null に設定します。

                disposedValue = true;
            }
        }

        // TODO: 上の Dispose(bool disposing) にアンマネージド リソースを解放するコードが含まれる場合にのみ、ファイナライザーをオーバーライドします。
        // ~InferenceServerManager() {
        //   // このコードを変更しないでください。クリーンアップ コードを上の Dispose(bool disposing) に記述します。
        //   Dispose(false);
        // }

        // このコードは、破棄可能なパターンを正しく実装できるように追加されました。
        public void Dispose()
        {
            // このコードを変更しないでください。クリーンアップ コードを上の Dispose(bool disposing) に記述します。
            Dispose(true);
            // TODO: 上のファイナライザーがオーバーライドされる場合は、次の行のコメントを解除してください。
            // GC.SuppressFinalize(this);
        }

        #endregion
    }
}