using System;
using System.IO;
using K = Keras.Backend;
using Keras;
using Keras.Models;
using Keras.Layers;
using Keras.Optimizers;
using tf = Tensorflow;
using static Tensorflow.Binding;
using tfb = Tensorflow.Binding;
using Tensorflow.NumPy;
using System.Runtime.Intrinsics.X86;

namespace ASR_proto
{

    class Tutorial
    {
        public static void Tensors_Constants()
        {
            var t1 = new tf.Tensor(3);
            var t2 = new tf.Tensor("Hello! Tensorflow.NET");
            var nd = new NDArray(new int[] { 3, 1, 1, 2 });
            var t3 = new tf.Tensor(nd);

            Console.WriteLine($"t1: {t1},\nt2: {t2},\nt3: {t3}\n");

            var nd1 = new NDArray(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            var tnd1 = new tf.Tensor(nd1);
            Console.WriteLine($"tnd1: {tnd1}");

            var c1 = tf.constant_op.constant(3);
            var c2 = tf.constant_op.constant(1.0f);
            var c3 = tf.constant_op.constant(2.0);

            Console.WriteLine($"c1: {c1}\nc2: {c2}\nc3: {c3}");

            var nd2 = np.array(new int[,]
            {
                {1, 2, 3},
                {4, 5, 6}
            });

            var tensor = tf.constant_op.constant(nd2);
            print($"tensor: {tensor}");
        }

        public static void Variables_Placeholders()
        {
            /// Disable eager
            //var x = tfb.tf.placeholder(tfb.tf.int32);
            //var y = x * 3;
            //using (var session = tfb.tf.Session())
            //{
            //    var result = session.run(y, feed_dict: new tf.FeedItem[]
            //    {
            //        new tf.FeedItem(x, 2)
            //    });
            //    Console.WriteLine($"y = {result}");
            //}
        }
    }
}