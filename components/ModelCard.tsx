import { useRef, useState, useEffect } from 'react';
import styles from '../styles/Home.module.css';
import * as tf from "@tensorflow/tfjs";

const ModelCard = (props: Props) => {

  const [label, setLabel] = useState("");
  const [infTime, setInfTime] = useState("");
  const [encoder, setEncoder] = useState(null);
  const [decoder, setDecoder] = useState(null);
  const [encBytes, setEncBytes] = useState(null);
  const [decBytes, setDecBytes] = useState(null);
  async function loadModelFromPath(path, setModel) {
    try {
      console.log(`Loading model from ${path}`);
      const model = await tf.loadGraphModel(path);
      setModel(model);
      console.log(`Finished loading model from ${path}`);
      console.log(model);
    }
    catch (err) {
      console.log(`Failed to load model from path ${path} with error: ${err}`);
      console.error(err);
    }
  }

  async function loadModels() {
    const encoderPath = "/tfjs/encoder_model/model.json";
    const decoderPath = "/tfjs/decoder_model_2/model.json";
    loadModelFromPath(encoderPath, setEncoder);
    loadModelFromPath(decoderPath, setDecoder);
  }

  async function inferenceMBART(): Promise<[any, number]> {
    console.log("loading model");
    const batchSize = 1;
    const inputIdsList = [250004, 1274, 2685, 903, 83, 10, 3034, 2]; 
    const seqLength = inputIdsList.length;
    const input_ids = tf.tensor(
      [inputIdsList],
      [batchSize, seqLength], 
      'int32'
    );
    const attention_mask = tf.ones(
      [batchSize, seqLength], 
      'int32',
    );
    const start = new Date();
    // const encoderStates = tf.zeros([1, 8, 1024], 'float32');
    let encoderStatesAsync;
    let decoderStatesAsync;
    const enc_profile = await tf.profile(() => {
      encoderStatesAsync = encoder.executeAsync(
        {input_ids: input_ids, attention_mask: attention_mask}
      );
    });
    const encoderStates = await encoderStatesAsync;
    console.log("ENCODER:");
    console.log(encoderStates);
    const dec_profile = await tf.profile(() => {
      decoderStatesAsync = decoder.executeAsync({
        input_ids: input_ids,
        encoder_hidden_states: encoderStates,
        encoder_attention_mask: attention_mask,
      });
    });
    const decoderStates = await decoderStatesAsync;
    const end = new Date();
    const inferenceTime = (end.getTime() - start.getTime())/1000;
    console.log(enc_profile);
    console.log(dec_profile);

    return [decoderStates, inferenceTime, enc_profile.peakBytes, dec_profile.peakBytes];
  }

  
  useEffect(() => {
    tf.ready().then(() => {
      console.log("tf ready; loading model now.");
      loadModels();
    });
  }, []);

  const performInference = async () => {
    var [infResult, infTime, encB, decB] = await inferenceMBART();

    setLabel(`Output shape: ${infResult[44].shape}`);
    setInfTime(`Inference speed: ${infTime} seconds`);
    setEncBytes(encB);
    setDecBytes(decB);

  };

  const runInference = () => { 
    // Clear out previous values.

    setLabel(`Doing Inference...`);
    setInfTime("");
   
    performInference();
  };

  const bytesToGB = (bytes) => {
    return Math.round(bytes*100/(2**30))/100;
  }

  return (
    <>
      {encoder == null && <p>Loading the encoder...</p>}
      {decoder == null && <p>Loading the decoder...</p>}
      {
        encoder != null && decoder != null && 
        <button
        className={styles.grid}
        onClick={runInference} >
        Run mBART
        </button>
      }
      <br/>
      <span>{label}</span>
      <span>{infTime}</span>
      {
        encBytes && decBytes && 
        <span>Peak memory usage: {bytesToGB(Math.max(encBytes, decBytes))}GB</span>
      }
    </>
  )

};

export default ModelCard;
