using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestDemo
{
    class LogUtil
    {
        /// <summary>
        /// log4net設定を読み込んで、必要なディレクトリを作成する
        /// </summary>
        public void MakeDirectoryForLogFile()
        {
            log4net.Repository.ILoggerRepository[] repositories = log4net.LogManager.GetAllRepositories();
            foreach (log4net.Repository.ILoggerRepository repository in repositories)
            {
                foreach (log4net.Appender.IAppender appender in repository.GetAppenders())
                {
                    log4net.Appender.FileAppender fileAppender = appender as log4net.Appender.FileAppender;
                    if (fileAppender != null)
                    {
                        // ログファイル
                        string logFile = fileAppender.File;
                        // ログファイルのパス
                        string path = System.IO.Path.GetDirectoryName(logFile);
                        // ディレクトリが存在しなければ作成
                        if (!System.IO.Directory.Exists(path))
                        {
                            System.IO.Directory.CreateDirectory(path);
                        }
                    }
                }
            }
        }

    }
}
