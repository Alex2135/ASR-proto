using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using Accord.Audio;
using Accord.DirectSound;
using Accord.Audio.Formats;

namespace ASR_proto
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Recorder recorder = new Recorder(() => Console.ReadKey(), "3.wav");
                recorder.RecordToAsync();
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);                
            }
        }

       /*
        * Тезисы:
        * Вступление.
        * 
        * Був розроблений проект застусунк на ЯП.
        * 
        * Цель которого реализовать многопоточноый
        * обмен данными.
        * 
        * Распознавание фиксированных фраз.
        * 
        * Какие авторы это реализовали, какие были варианты.
        * 
        * Библиотеки на которых будет реализация.
        * 
        * Дальнейшая реализация проекта. Где может быть
        * использована: школы, ВУЗы, музей науки.
        * 
        * актуально во время ковида, потому что безконтактное
        * взаимодействие с компьютером.
        * 
        * Рассказать про нейронную сеть и схему нейронной сети.
        * 
        * Подать до 25.03
        */


    }
}
