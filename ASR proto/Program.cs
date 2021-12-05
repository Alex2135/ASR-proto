using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Spectrogram;
using CsvHelper;
using CsvHelper.Expressions;
using CsvHelper.TypeConversion;
using CsvHelper.Configuration;
using System.Globalization;

namespace ASR_proto
{
    [Serializable]
    public class SpeechData
    {
        public string client_id { get; set; }
        public string path { get; set; }
        public string sentence { get; set; }
        public int up_votes { get; set; }
        public int down_votes { get; set; }
        public string age { get; set; }
        public string gender { get; set; }
        public string accent { get; set; }
        public string locale { get; set; }
        public string segment { get; set; }

        public string ImagePath
        {
            get
            {
                string imgPath = Path.Combine(DataPreparation.spectrogramsPath, path);
                int lastPointIndex = imgPath.LastIndexOf('.');
                imgPath = imgPath.Remove(lastPointIndex) + ".jpg";
                return imgPath;
            }
        }
    }

    class Program
    {
        public static void ConvertMP3sToSpectrogram(
            string dataDir = @"D:\ML\Speech recognition\NLP_diploma\uk",
            string clipsDir = "clips",
            string spectroDir = "test_spectrograms",
            string dataSourceFilePath = "test.tsv")
        {
            string path = dataDir;
            DataPreparation.clipsPath = Path.Combine(path, clipsDir);
            DataPreparation.spectrogramsPath = Path.Combine(path, spectroDir);
            path = Path.Combine(path, dataSourceFilePath);
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            List<SpeechData> trainData = new List<SpeechData>();
            var preparator = new DataPreparation();

            // Get data from .tsv file
            Console.WriteLine($"Loading data from: {path}");
            using (var _reader = new StreamReader(path))
            {

                CsvConfiguration myConfig = new CsvConfiguration(CultureInfo.InvariantCulture);
                myConfig.Delimiter = "\t";
                myConfig.Encoding = System.Text.Encoding.UTF8;
                using (var csvReader = new CsvReader(_reader, myConfig))
                {
                    while (csvReader.Read())
                    {
                        var data = csvReader.GetRecord<SpeechData>();
                        preparator.Prepare(ref data);
                        trainData.Add(data);
                        //break;
                    }
                }
            }
            Console.WriteLine("Train data was loaded!");
            Console.WriteLine(trainData.Count);

            bool isFormedSpectrograms = false;
            if (!isFormedSpectrograms)
            {
                int counter = 0;
                foreach (var sample in trainData)
                {
                    preparator.GenerateSpectrogram(sample);
                    counter++;
                    if (counter % 400 == 0)
                        Console.WriteLine($"{counter} audio files processed");
                }
                Console.WriteLine("Spectrograms data was created!");
            }
        }

        public static void RecordAudioFromMicro(
            string audioFileName = "vlog.wav",
            string imageFilePath = "images\\viridis.jpg")
        {
            string path = audioFileName;
            Recorder recorder = new Recorder(null, path);
            Console.WriteLine("Start recording...");
            recorder.FinishRecord = () =>
            {
                Console.ReadKey();
            };
            recorder.RecordAudioToFile();
            while (recorder.IsRecordContinue) { }
            Console.WriteLine("Finish recording...");
            SpectrogramBuilder spectrogramBuilder = new SpectrogramBuilder();
            spectrogramBuilder.ImagePath = imageFilePath;
            spectrogramBuilder.ImageColormap = Colormap.GrayscaleReversed; 
            spectrogramBuilder.BuildSpectrogram(path);
            Console.WriteLine($"Record save to: \"{audioFileName}\"");
        }

        static string GetRandomString(int length)
        {
            var chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            var stringChars = new char[length];
            var random = new Random();

            for (int i = 0; i < stringChars.Length; i++)
            {
                stringChars[i] = chars[random.Next(chars.Length)];
            }

            var finalString = new String(stringChars);
            return finalString;
        }

        static void Main(string[] args)
        {            
            try
            {

                /*
                 * 0. None
                 * 1. "Запустити відео обтікання будинків"
                    2. "запустити відео демонстрації вихорів"
                    3. "запустити відео обтікання крила"
                    4. "закрити відео
                 */

                const string DATA_DIR = @"D:\ML\Speech recognition\NLP_diploma\uk";
                const string CLIPS_DIR = @"D:\ML\Speech recognition\NLP_diploma\uk\clips_classifire\aug+orig";//@"D:\ML\Speech recognition\NLP_diploma\uk\clips";
                const string SPECTRO_DIR = @"D:\ML\Speech recognition\NLP_diploma\uk\mel_spectrograms_classifire";//@"D:\ML\Speech recognition\NLP_diploma\uk\train_spectrograms";//

                DataPreparation.clipsPath = CLIPS_DIR;
                DataPreparation.spectrogramsPath = SPECTRO_DIR;

                string[] audios = Directory.GetFiles(CLIPS_DIR, "*.wav");

                foreach (string path in audios)
                {
                    DataPreparation.GenerateSpectrogram(path);
                    Console.WriteLine(path);
                }

                /*
                string fileName = $"4_{GetRandomString(12)}";

                RecordAudioFromMicro(
                    audioFileName: Path.Combine(DATA_DIR, $"clips_classifire\\{fileName}.wav"),
                    imageFilePath: Path.Combine(DATA_DIR, $"spectrograms_classifire\\{fileName}.jpg")
                ) ;
                */

                //ConvertMP3sToSpectrogram();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}

