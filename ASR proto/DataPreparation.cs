using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Spectrogram;

namespace ASR_proto
{
    class DataPreparation
    {
        public static string clipsPath = "";
        public static string spectrogramsPath = "";

        public void Prepare(ref SpeechData _sample)
        {
            if (clipsPath == "") throw new Exception("Clips path not set");

            Char[] specialChars = new Char[] { '!', ',', '.', '-', '?', '–', ' ' };
            var arr = _sample.sentence.ToLower().Split(specialChars);
            string result = "";
            for (int i = 0; i < arr.Length; i++)
                if (arr[i] != "") result += arr[i] + " ";
            _sample.sentence = result.Trim();
        }

        public void GenerateSpectrogram(SpeechData _sample)
        {
            string clipPath = Path.Combine(clipsPath, _sample.path);
            string imgPath = _sample.ImagePath;

            SpectrogramBuilder spectrogramBuilder = new SpectrogramBuilder();
            spectrogramBuilder.ImagePath = imgPath;
            spectrogramBuilder.ImageColormap = Colormap.Viridis;
            spectrogramBuilder.BuildSpectrogram(clipPath);
        }
    }

}
