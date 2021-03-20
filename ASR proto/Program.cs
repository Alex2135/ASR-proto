using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using Accord.Audio;
using Accord.DirectSound;
using Accord.Audio.Formats;

namespace ASR_proto
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Recorder recorder = new Recorder(() => Console.ReadKey(), "3.wav");
                recorder.RecordToAsync();
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);                
            }
        }
    }
}
