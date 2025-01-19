import React from 'react'
import { fadeIn, textVariantPhone } from '../constants/motion'
import { Tilt } from 'react-tilt'
import { certifications, mlAlgos } from '../constants/info'
import SectionWrapper from './SectionWrapper'
import { styles } from '../constants/styles'
import { motion } from 'framer-motion'

const ServiceCardAlt = ({index, name, icon, certificate, link}) => {
    return (
      <Tilt className = "w-[250px]">
        <motion.div 
          variants={fadeIn("right", "spring", 0.15 * index, 0.75)}
          className='w-[250px] green-pink-gradient p-[1px] rounded-[20px] shadow-card'
        >
          <div 
            className="bg-tertiary rounded-[20px] py-5 px-6 min-h-[130px] flex justify-evenly items-center flex-col"
            options = {{
              max: 45,
              scale: 1,
              speed: 1000
            }}
          >
            <img src={icon} alt='title'className='w-12 h-12 object-contain'/>
            <h3 className="text-white font-semibold text-center text-sm">
              {name}
            </h3>
            <a href={link} target='_blank'>
              <img src = {certificate} alt='title' className='object-contain w-40 h-32'/>
            </a>
          </div>
        </motion.div>
      </Tilt>
    )
  }

const Certifications = () => {
    return (
        <>
            <motion.div variants={textVariantPhone()}>
                <h2 className={styles.sectionHeadText}>Certifications.</h2>
            </motion.div>
            <div className="mt-10 flex flex-wrap gap-10">
                {certifications.map((skill, index)=>{
                    return (
                    <ServiceCardAlt key = {skill.name} index = {index} {...skill} />
                    )
                })}
            </div>
        </>
    )
}

export default SectionWrapper(Certifications, "skills");