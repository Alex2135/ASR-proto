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
        public string ImagePath { get; set; }
        public int FFTSize { get; set; }
        public int FFTStepSize { get; set; }
        public double MaxFrequency { get; set; }
        public Colormap ImageColormap { get; set; }

        public SpectrogramBuilder()
        {
            ImagePath = "1.jpg";
            FFTSize = 8192;
            FFTStepSize = 256;
            MaxFrequency = 11000;
            ImageColormap = Colormap.Blues;
        }
        public SpectrogramBuilder(string _path)
        {
            ImagePath = _path;
            FFTSize = 8192;
            FFTStepSize = 256;
            MaxFrequency = 11000;
            ImageColormap = Colormap.Blues;
        }

        public void BuildSpectrogram(string _wavPath)
        {
            double[] audio; int sampleRate;
            (audio, sampleRate) = ReadWAV(_wavPath);

            var sg = new SpectrogramGenerator(sampleRate, fftSize: FFTStepSize, stepSize: FFTStepSize, maxFreq: MaxFrequency);
            sg.Add(audio);
            sg.SetColormap(ImageColormap);
            sg.SaveImage(ImagePath);
        }

        public (double[] audio, int sampleRate) ReadWAV(string filePath, double multiplier = 16_000)
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
