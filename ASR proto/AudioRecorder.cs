using System;
using System.IO;
using System.Threading;
using System.Collections.Generic;
using System.Threading.Tasks;
using Accord.Audio;
using Accord.DirectSound;
using Accord.Audio.Formats;

namespace ASR_proto
{
    public class Recorder
    {
        public Action FinishRecord { get; set; }
        public string RecordPath { get; set; }
        public bool IsRecordContinue { get; set; }
        private FileStream _fileStream;
        private AudioCaptureDevice _microphone;
        private WaveEncoder _audioSaver;

        public Recorder(Action _finishRecord, string _path = "1.wav")
        {
            FinishRecord = _finishRecord;
            RecordPath = _path;
            IsRecordContinue = false;
        }

        private void source_NewFrame(object sender, NewFrameEventArgs eventArgs)
        {
            // Read current frame...
            Signal s = eventArgs.Signal;
            // encode frame to file
            _audioSaver.Encode(s);
        }

        public void SetRecorder(Guid? guid = null)
        {
            Guid? deviceGUID = guid;
            var capture = new AudioDeviceCollection(AudioDeviceCategory.Capture);
            foreach (var d in capture)
            {
                //Console.WriteLine($"{d.Guid}: {d.Description}");
                if (d.Description.IndexOf("Audio") != -1)
                {
                    deviceGUID = d.Guid;
                    break;
                }
            }
            Guid _guid = deviceGUID ?? throw new Exception("Guid of capture devise is not set");
            _microphone = new AudioCaptureDevice(_guid);
            _microphone.SampleRate = 44100;
            _microphone.Channels = 2;
            _microphone.Format = SampleFormat.Format32BitIeeeFloat;
            _microphone.NewFrame += source_NewFrame;
        }

        private async void OnListeningAsync()
        {
            Console.WriteLine("Press key to stop this shit!");
            // Wait until operations end
            var finishEventTask = new Task(FinishRecord);
            finishEventTask.Start();
            await finishEventTask;
            Thread.Sleep(500); // For capturing last signals
            _microphone.SignalToStop();
        }

        public async void RecordToAsync(string _path = "1.wav")
        {
            if (_microphone == null) SetRecorder();
            if (FinishRecord == null) throw new Exception("FinishRecord function not set!");
            if (_path == null) throw new ArgumentNullException("Error! Argument path is null!");
            else if (_path != "1.wav") RecordPath = _path;


            Task t = new Task(OnListeningAsync);
            try
            {
                _fileStream = new FileStream(RecordPath, FileMode.OpenOrCreate, FileAccess.Write);
                _audioSaver = new WaveEncoder();
                _audioSaver.Open(_fileStream);
                IsRecordContinue = true;
                _microphone.Start();
                t.Start();
                await t;
                _microphone.WaitForStop();
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
            finally
            {
                t.Dispose();
                _microphone.Dispose();
                _audioSaver?.Close();
                _fileStream?.Close();
                IsRecordContinue = false;
            }
        }
    }
}
