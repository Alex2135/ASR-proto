using System;
using Spectrogram;

namespace ASR_proto
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                string path = "3.wav";
                Recorder recorder = new Recorder(null, path);
                recorder.FinishRecord = () =>
                {
                    Console.ReadKey();
                };
                recorder.RecordAudioToFile();

                while (recorder.IsRecordContinue) { }

                SpectrogramBuilder spectrogramBuilder = new SpectrogramBuilder();
                spectrogramBuilder.ImagePath = "images\\viridis.jpg";
                spectrogramBuilder.ImageColormap = Colormap.Viridis;
                spectrogramBuilder.BuildSpectrogram(path);

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}
