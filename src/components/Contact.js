import React, {useState, useRef} from 'react';
import { motion } from 'framer-motion';
import emailjs from "@emailjs/browser";

import { styles } from '../constants/styles';
import SectionWrapper from './SectionWrapper';
import { slideIn } from '../constants/motion';

const Contact = () => {
  const formRef = useRef();
  const [form, setForm] = useState({
    name: "",
    email: "",
    message: "",
  });
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const {name, value} = e.target;
    setForm({...form, [name]: value});
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);

    emailjs.send(
      "service_tp1qft6", 
      "template_6aoiz3r", 
      {
        from_name: form.name,
        to_name: "Karandeep Shoker",
        from_email: form.email,
        to_email: "karandeep.shoker@outlook.com",
        message: form.message,
      },
      "hlSVlzxpw2XrnfJ-v"
    ).then(()=>{
      setLoading(false);
      alert("Thank you. I will get back to you as soon as possible.");
      setForm({
        name: "",
        email: "",
        message: ""
      })
    }, (error)=>{
      setLoading(false);
      console.log(error);

      alert("Something went wrong.")
    });

  }



  return (
    <div className="xl:mt-6 xl:flex-row flex-col-reverse flex gap-10 ">
      <div
        className='flex-1 bg-black-100 p-8 rounded-2xl'
        data-aos = "fade-right"
        aos-delay = {100}
      >
        <p className={styles.sectionSubText}>Get in touch</p>
        <h3 className={styles.sectionHeadText}>Contact.</h3>

        <form 
          ref={formRef}
          onSubmit={handleSubmit}
          className='flex mt-12 flex-col gap-8'
        >
          <label htmlFor="" className="flex flex-col">
            <span className='text-white'>Your Name</span>
            <input
              type='text'
              name='name'
              value={form.name}
              onChange={handleChange}
              placeholder="What's your name?"
              className='bg-tertiary py-4 px-6 placeholder:text-secondary text-white rounded-lg outline-none border-none font-medium'
            />
          </label>
          <label htmlFor="" className="flex flex-col">
            <span className='text-white'>Your Email</span>
            <input
              type='email'
              name='email'
              value={form.email}
              onChange={handleChange}
              placeholder="What's your Email?"
              className='bg-tertiary py-4 px-6 placeholder:text-secondary text-white rounded-lg outline-none border-none font-medium'
            />
          </label>
          <label htmlFor="" className="flex flex-col">
            <span className='text-white'>Your Message</span>
            <textarea
              rows={7}
              name='message'
              value={form.message}
              onChange={handleChange}
              placeholder="What do you want to say?"
              className='bg-tertiary py-4 px-6 placeholder:text-secondary text-white rounded-lg outline-none border-none font-medium'
            />
          </label>

          <button
            type='submit'
            className='bg-tertiary py-3 px-8 outline-none w-fit text-white font-bold shadow-md shadow-primary rounded-xl'
          >
            {loading ? "Sending..." : "Send"}
          </button>
        </form>
      </div>
    </div>
  )
}

export default SectionWrapper(Contact, "contact");