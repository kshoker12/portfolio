import React, { Component } from 'react'
import { motion } from 'framer-motion'
import { styles } from '../constants/styles'
import { staggerContainer } from '../constants/motion'

const SectionWrapper = (Component, idName) => function HOC() {
    return (
        <motion.section
            variants={staggerContainer()}
            initial = "hidden"
            viewport={{ once: true, amount: 0.25}}
            whileInView="show"
            className={`${styles.padding} max-w-7xl mx-auto relative z-0`}
        >
            <span className="hash-span" id={idName}>&nbsp;</span>
            <Component/>
        </motion.section>
    );
};

export default SectionWrapper