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
    class Program
    {
        [Serializable]
        public class TrainData
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
        }

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
                path = Path.Combine(path, "train.tsv");
                Console.OutputEncoding = System.Text.Encoding.UTF8;
                List<TrainData> trainData = new List<TrainData>();

                using (var _reader = new StreamReader(path))
                {

                    CsvConfiguration myConfig = new CsvConfiguration(CultureInfo.InvariantCulture);
                    myConfig.Delimiter = "\t";
                    myConfig.Encoding = System.Text.Encoding.UTF8;
                    using (var csvReader = new CsvReader(_reader, myConfig))
                    {
                        while (csvReader.Read())
                        { 
                            var data = csvReader.GetRecord<TrainData>();
                            trainData.Add(data);
                            
                            Console.WriteLine($"{data.sentence}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}

