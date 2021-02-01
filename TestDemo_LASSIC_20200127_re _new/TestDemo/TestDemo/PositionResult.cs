using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestDemo
{
    /// <summary>
    /// 喉位置データの受け取り用
    /// </summary>
    class PositionResult
    {
        /// <summary>
        /// x
        /// </summary>
        [JsonProperty("x")]
        public string X { get; set; }
        
        /// <summary>
        /// y
        /// </summary>
        [JsonProperty("y")]
        public string Y { get; set; }
        
    }
}
