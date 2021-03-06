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
        static void Main(string[] args)
        {
            try
            {

                //string path = "3.wav";
                //Recorder recorder = new Recorder(null, path);
                //recorder.FinishRecord = () =>
                //{
                //    Console.ReadKey();
                //};
                //recorder.RecordAudioToFile();
                //while (recorder.IsRecordContinue) { }
                //SpectrogramBuilder spectrogramBuilder = new SpectrogramBuilder();
                //spectrogramBuilder.ImagePath = "images\\viridis.jpg";
                //spectrogramBuilder.ImageColormap = Colormap.Viridis;
                //spectrogramBuilder.BuildSpectrogram(path);

                string path = @"D:\ML\Speech recognition\NLP_diploma\uk";
                DataPreparation.clipsPath = Path.Combine(path, "clips");
                DataPreparation.spectrogramsPath = Path.Combine(path, "train_spectrograms");
                path = Path.Combine(path, "train.tsv");
                Console.OutputEncoding = System.Text.Encoding.UTF8;
                List<SpeechData> trainData = new List<SpeechData>();
                var preparator = new DataPreparation();

                // Get data from train.tsv
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
                            break;
                        }
                    }
                }
                Console.WriteLine("Train data was loaded!");
                Console.WriteLine(trainData.Count);

                bool isFormedSpectrograms = true;
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
                    Console.WriteLine("Spectrograms train data was created!");
                }

                TFModel model = new TFModel(trainData);
                model.Train();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}

