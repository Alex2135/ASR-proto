using System;
using System.Collections.Generic;
using NAudio;
using Spectrogram;
using System.Linq;

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
                recorder.RecordToAsync();

                while (recorder.IsRecordContinue) { }

                double[] audio; int sampleRate;
                (audio, sampleRate) = ReadWAV(path);

                var sg = new SpectrogramGenerator(sampleRate, fftSize: 8192, stepSize: 200, maxFreq: 11000);
                sg.Add(audio);
                sg.SetColormap(Colormap.Grayscale);
                sg.SaveImage($"gray.png");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

        }

        public static (double[] audio, int sampleRate) ReadWAV(string filePath, double multiplier = 16_000)
        {
            using var afr = new NAudio.Wave.AudioFileReader(filePath);
            int sampleRate = afr.WaveFormat.SampleRate;
            int sampleCount = (int)(afr.Length / afr.WaveFormat.BitsPerSample / 8);
            int channelCount = afr.WaveFormat.Channels;
            var audio = new List<double>(sampleCount);
            var buffer = new float[sampleRate * channelCount];
            int samplesRead = 0;
            while ((samplesRead = afr.Read(buffer, 0, buffer.Length)) > 0)
                audio.AddRange(buffer.Take(samplesRead).Select(x => x * multiplier));
            return (audio.ToArray(), sampleRate);
        }
    }
}
