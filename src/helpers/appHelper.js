import Aos from "aos";

/**
 * @description Initialize the Animation on start library to allow animated components
 */
const initializeAOS = () => {
    Aos.init({
        duration: 2000
      });
    Aos.refresh();
};

export { initializeAOS };