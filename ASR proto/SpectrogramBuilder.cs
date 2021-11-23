using System;
using System.Collections.Generic;
using System.Text;
using System.Collections.Generic;
using NAudio;
using System.Linq;
using Spectrogram;

namespace ASR_proto
{
    class SpectrogramBuilder
    {
        public string ImagePath { get; set; } = "1.jpg";
        public int FFTSize { get; set; } = 4096; // 4096
        public int FFTStepSize { get; set; } = 512;
        public double MaxFrequency { get; set; } = 5_000;
        public Colormap ImageColormap { get; set; } = Colormap.Grayscale;
        public int SpectrogramWidth { get; set; } = (int)Math.Pow(2, 9);

        public SpectrogramBuilder()
        {
        }

        public SpectrogramBuilder(string _path)
        {
            ImagePath = _path;
        }

        public void BuildSpectrogram(string _wavPath)
        {
            double[] audio; int sampleRate;
            (audio, sampleRate) = ReadWAV(_wavPath);

            var sg = new SpectrogramGenerator(sampleRate, fftSize: FFTSize, stepSize: FFTStepSize, maxFreq: MaxFrequency);
            //sg.SetFixedWidth(SpectrogramWidth);
            sg.Add(audio);
            sg.SetColormap(ImageColormap);
            sg.SaveImage(ImagePath);
        }

        public (double[] audio, int sampleRate) ReadWAV(string filePath)
        {
            double multiplier = MaxFrequency;
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
