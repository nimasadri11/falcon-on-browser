import type { NextPage } from 'next';
import Head from 'next/head';
import styles from '../styles/Home.module.css';
import ModelCard from "../components/ModelCard";


const Home: NextPage = () => {  
  
  return (
    <div className={styles.container}>
      <main className={styles.main}>
        <h1 className={styles.title}>
          mBART
        </h1>
      <ModelCard/>
      <div id="result" className="mt-3">
      </div>
      </main>

    </div>
  )
}

export default Home
