using System;
using System.Collections.Generic;
using Accord.Imaging.Formats;
using System.Drawing;
using System.Text;
using K = Keras.Backend;
using Keras;
using Keras.Models;
using Keras.Layers;
using Keras.Optimizers;
using static Tensorflow.Binding;
using Tensorflow.NumPy;

namespace ASR_proto
{
    class TFModel
    {
        private List<SpeechData> trainData;

        public TFModel(List<SpeechData> _trainData)
        {
            trainData = _trainData;
        }

        public NDArray ImageToArray(string _path, int _preferWidth, int _preferHeight)
        {
            var image = Bitmap.FromFile(_path) as Bitmap;
            Bitmap buf = new Bitmap(_preferWidth, _preferHeight);
            var result = np.zeros(new int[]{ _preferWidth, _preferHeight});

            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    Color c = image.GetPixel(i, j);
                    int grey = (int)(0.299 * c.R + 0.587 * c.G + 0.114 * c.B);
                    buf.SetPixel(i, j, Color.FromArgb(255, grey, grey, grey));
                    result[i][j] = (grey / 255d);
                }
            }

            //var spath = _path + "_HUI.jpg";
            //buf.Save(spath);
            //Console.WriteLine(spath);
            result = result.reshape((_preferHeight, _preferWidth));
            return result;
        }

        public void Train()
        {
            tf.compat.v1.disable_eager_execution();

            var sample = trainData[0];
            int imgWidth = 1024;
            int imgHeight = 256;

            var img = ImageToArray(sample.ImagePath, imgWidth, imgHeight);

            Console.WriteLine($"{img.shape[0]}, {img.shape[1]}");

            var model = new Sequential();
            model.Add(new Input(shape: (512, 512)));
            model.Add(new Conv1D(
                8, // Filters 
                4,  // Kernel size
                2,  // Strides
                activation: "relu")
            );
            model.Add(new Conv1D(
                8, // Filters 
                4,  // Kernel size
                2,   // Strides
                activation: "relu")
            );
            //model.Add(new Lambda())



            //var x = tf.Variable(10, name: "x");
            //using (var session = tf.Session())
            //{
            //    session.run(x.initializer);
            //    var result = session.run(x);
            //    Console.WriteLine((int)result);
            //}
        }
    }
}
