import HeroSlider from '../components/HeroSlider';
import ServicesGrid from '../components/ServiceGrid';
import Footer from '../components/Footer';

function Home() {
  return (
    <div className="bg-gray-100 min-h-screen">
      <HeroSlider />
      <ServicesGrid />
      <Footer />
    </div>
  );
}

export default Home;